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

  // Parse user responses
  const userResponses = JSON.parse(query);

  // Store analysis results
  let imageAnalysis: string | null = null;
  let combinedAnalysis = "";

  // Define the graph state
  const GraphState = Annotation.Root({
    messages: Annotation<BaseMessage[]>({
      reducer: (x, y) => x.concat(y),
    }),
    imageAnalysis: Annotation<string | null>({
      reducer: (_, y) => y,
    }),
    userResponses: Annotation<any>({
      reducer: (_, y) => y,
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
  const toolNode = new ToolNode<typeof GraphState.State>(tools);

  const model = new ChatOpenAI({
    modelName: "gpt-4o-mini",
    temperature: 0,
  }).bindTools(tools);

  const vision = new ChatAnthropic({
    model: "claude-3-opus-20240229",
  });

  if (image_path) {
    console.log("Processing image analysis...");
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
    imageAnalysis = String(response.content);
    console.log("Image analysis completed:", imageAnalysis);
  }

  // Define the function that determines whether to continue or not
  function shouldContinue(state: typeof GraphState.State) {
    const messages = state.messages;
    const lastMessage = messages[messages.length - 1] as AIMessage;

    if (lastMessage.tool_calls?.length) {
      return "tools";
    }
    return "__end__";
  }

  // Define the function that calls the model
  async function callModel(state: typeof GraphState.State) {
    const prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        `You are an AI Skincare Expert specializing in providing personalized skincare recommendations. Your task is to analyze both the user's questionnaire responses and image analysis (if available) to provide tailored product recommendations.

Current Context:
- User Questionnaire: {user_responses}
- Image Analysis: {image_analysis}

Based on this information:
1. Analyze both the questionnaire responses and image analysis
2. Identify the key skincare needs and concerns
3. Use the product lookup tool to find suitable products
4. Provide a comprehensive recommendation with explanation

Format your response as follows:
1. Brief analysis of the user's skin condition and needs
2. Product recommendations with explanations for each
3. Usage instructions and any additional advice

You have access to the following tools: {tool_names}
Current time: {time}`,
      ],
      new MessagesPlaceholder("messages"),
    ]);

    const formattedPrompt = await prompt.formatMessages({
      user_responses: JSON.stringify(userResponses, null, 2),
      image_analysis: imageAnalysis || "No image analysis available",
      tool_names: tools.map((tool) => tool.name).join(", "),
      time: new Date().toISOString(),
      messages: state.messages,
    });

    const result = await model.invoke(formattedPrompt);
    return { messages: [result] };
  }

  const workflow = new StateGraph(GraphState)
    .addNode("agent", callModel)
    .addNode("tools", toolNode)
    .addEdge("__start__", "agent")
    .addConditionalEdges("agent", shouldContinue)
    .addEdge("tools", "agent");

  const checkpointer = new MongoDBSaver({ client, dbName });
  const app = workflow.compile({ checkpointer });

  const finalState = await app.invoke(
    {
      messages: [new HumanMessage("Please provide skincare recommendations based on my responses and image.")],
      imageAnalysis: imageAnalysis,
      userResponses: userResponses,
    },
    { recursionLimit: 15, configurable: { thread_id: thread_id } }
  );

  return finalState.messages[finalState.messages.length - 1].content;
}
