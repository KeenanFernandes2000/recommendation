import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { ChatAnthropic } from "@langchain/anthropic";
import { AIMessage, BaseMessage, HumanMessage, SystemMessage } from "@langchain/core/messages";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { StateGraph } from "@langchain/langgraph";
import { Annotation } from "@langchain/langgraph";
import { tool } from "@langchain/core/tools";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { MongoDBSaver } from "@langchain/langgraph-checkpoint-mongodb";
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb";
import { MongoClient } from "mongodb";
import { z } from "zod";
import "dotenv/config";
import fs from "fs";

export async function callAgent(
  client: MongoClient,
  query: string,
  thread_id: string,
  image_path: string | null
) {
  // Define the MongoDB database and collection
  const dbName = "prod_data";
  const db = client.db(dbName);
  const collection = db.collection("products");

  // Define the graph state
  const GraphState = Annotation.Root({
    messages: Annotation<BaseMessage[]>({
      reducer: (x, y) => x.concat(y),
    }),
  });

  // Define the tools for the agent to use
  const prodLookupTool = tool(
    async ({ query, n = 4 }) => {
      console.log("Product lookup tool called");

      const dbConfig = {
        collection: collection,
        indexName: "vector_index",
        textKey: "embedding_text",
        embeddingKey: "embedding",
      };

      // Initialize vector store
      const vectorStore = new MongoDBAtlasVectorSearch(
        new OpenAIEmbeddings(),
        dbConfig
      );

      const result = await vectorStore.similaritySearchWithScore(query, n);
      return JSON.stringify(result);
    },
    {
      name: "product_lookup",
      description: "Gathers product details from the prod database",
      schema: z.object({
        query: z.string().describe("The search query"),
        n: z
          .number()
          .optional()
          .default(10)
          .describe("Number of results to return"),
      }),
    }
  );

  const tools = [prodLookupTool];

  // We can extract the state typing via `GraphState.State`
  const toolNode = new ToolNode<typeof GraphState.State>(tools);

  const model = new ChatOpenAI({
    modelName: "gpt-4o-mini",
    temperature: 0,
  }).bindTools(tools);

  const vision = new ChatAnthropic({
    model: "claude-3-5-sonnet-20241022",
  });


  if (image_path) {
    console.log("Image path provided");
    const file = fs.readFileSync(image_path);

    const base64_image = Buffer.from(file).toString("base64");

    const system_message = new SystemMessage({
      content: "You are an AI Dermatologist and Skincare Expert specializing in providing tailored skincare advice based on image analysis. Your task is to analyze the provided image description and generate a structured response detailing your findings.",
    });

    const message = new HumanMessage({
      content: [
        {
          type: "text",
          text: `You are an AI Dermatologist and Skincare Expert specializing in providing tailored skincare advice based on image analysis. Your task is to analyze the provided image description and generate a structured response detailing your findings.

Please follow these steps to complete your analysis:

1. Carefully review the image.
2. Use your expertise to determine the skin condition, skin type, visible conditions, severity, and affected areas based on the information given.
3. Before formulating your final response, wrap your observations and reasoning inside <dermatological_assessment> tags. This step is crucial for ensuring a thorough and accurate assessment. In this section:
   - List out key observations from the image analysis, categorizing them into skin type, visible conditions, severity, and affected areas.
   - Consider potential alternative diagnoses and explain why they were ruled out.
   - It's OK for this section to be quite long.
4. After your assessment, provide your findings in the specified JSON format.

Important Instructions:
- Always provide a diagnosis based on the image analysis.
- Strictly adhere to the requested output format.
- Do not include any additional text or explanations outside of the JSON structure.

The required output format is as follows:

{
  "skin_analysis": {
    "skin_type": "",
    "visible_conditions": {
      "primary": "",
      "secondary": "",
      "other_observations": []
    }
  },
  "severity": "",
  "affected_areas": []
}

Please ensure that all fields are filled appropriately based on your analysis. If a field is not applicable, use an empty string or empty array as appropriate.

Begin your response with your dermatological assessment, followed by the JSON output.`,
        },
        {
          type: "image_url",
          image_url: { url: "data:image/jpeg;base64," + base64_image },
        },
      ],
    });
    const response = await vision.invoke([system_message, message]);
    console.log(response.content);
  }

  // Define the function that determines whether to continue or not
  function shouldContinue(state: typeof GraphState.State) {
    const messages = state.messages;
    const lastMessage = messages[messages.length - 1] as AIMessage;

    // If the LLM makes a tool call, then we route to the "tools" node
    if (lastMessage.tool_calls?.length) {
      return "tools";
    }
    // Otherwise, we stop (reply to the user)
    return "__end__";
  }

  // Define the function that calls the model
  async function callModel(state: typeof GraphState.State) {
    const prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        `You are a helpful AI assistant, collaborating with other assistants. Use the provided tools to progress towards answering the question. If you are unable to fully answer, that's OK, another assistant with different tools will help where you left off. Execute what you can to make progress. If you or any of the other assistants have the final answer or deliverable, prefix your response with FINAL ANSWER so the team knows to stop. You have access to the following tools: {tool_names}.\n{system_message}\nCurrent time: {time}.`,
      ],
      new MessagesPlaceholder("messages"),
    ]);

    const formattedPrompt = await prompt.formatMessages({
      system_message: "You are an AI Dermatologist and Skincare Expert specializing in providing tailored skincare advice and product recommendations. Use your knowledge of dermatology and skincare to understand the user's concerns, diagnose potential skin conditions, and recommend suitable products or routines. Always aim to give evidence-based and personalized advice that enhances the user’s skincare journey.",
      time: new Date().toISOString(),
      tool_names: tools.map((tool) => tool.name).join(", "),
      messages: state.messages,
    });

    const result = await model.invoke(formattedPrompt);

    return { messages: [result] };
  }

  // Define a new graph
  const workflow = new StateGraph(GraphState)
    .addNode("agent", callModel)
    .addNode("tools", toolNode)
    .addEdge("__start__", "agent")
    .addConditionalEdges("agent", shouldContinue)
    .addEdge("tools", "agent");

  // Initialize the MongoDB memory to persist state between graph runs
  const checkpointer = new MongoDBSaver({ client, dbName });

  // This compiles it into a LangChain Runnable.
  // Note that we're passing the memory when compiling the graph
  const app = workflow.compile({ checkpointer });

  // Use the Runnable
  const finalState = await app.invoke(
    {
      messages: [new HumanMessage(query)],
    },
    { recursionLimit: 15, configurable: { thread_id: thread_id } }
  );

  // console.log(JSON.stringify(finalState.messages, null, 2));
  console.log(finalState.messages[finalState.messages.length - 1].content);

  return finalState.messages[finalState.messages.length - 1].content;
}
