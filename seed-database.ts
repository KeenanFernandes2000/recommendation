import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { StructuredOutputParser } from "@langchain/core/output_parsers";
import { MongoClient } from "mongodb";
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb";
import { z } from "zod";
import "dotenv/config";

const client = new MongoClient(process.env.MONGODB_ATLAS_URI as string);
const llm = new ChatOpenAI({
  modelName: "gpt-4o-mini",
  temperature: 0.7,
});

const HealthCareProductSchema = z.object({
  product_id: z.string(), // Unique identifier for the product
  product_name: z.string(), // Name of the product
  category: z.string(), // Category like hair care, skin care, etc.
  description: z.string(), // Brief description of the product
  ingredients: z.array(z.string()), // List of ingredients
  usage_instructions: z.string(), // How to use the product
  price: z.object({
    amount: z.number(), // Price amount
    currency: z.string(), // Currency of the price
  }),
  stock: z.number(), // Quantity of stock available
  manufacturer_details: z.object({
    name: z.string(), // Manufacturer's name
    address: z.string(), // Manufacturer's address
    country: z.string(), // Country of manufacturing
  }),
  ratings: z.array(
    z.object({
      rating_date: z.string(), // Date when the rating was given
      rating: z.number(), // Rating value
      comments: z.string().optional(), // Optional comments from the user
    })
  ),
  benefits: z.array(z.string()), // List of benefits of the product
  precautions: z.string().optional(), // Precautions or warnings, if any
  related_products: z.array(z.string()).optional(), // List of related products by ID
  availability: z.object({
    online: z.boolean(), // Whether it's available for online purchase
    in_store: z.boolean(), // Whether it's available in-store
  }),
  expiration_date: z.string().optional(), // Expiration date, if applicable
  notes: z.string().optional(), // Additional notes or information
});

type Product = z.infer<typeof HealthCareProductSchema>;

const parser = StructuredOutputParser.fromZodSchema(
  z.array(HealthCareProductSchema)
);

async function generateSyntheticData(): Promise<Product[]> {
  const prompt = `You are a helpful assistant that generates healthcare product data. Generate 10 fictional healthcare product records. Each record should include the following fields: product_id, product_name, category, description, ingredients, usage_instructions, price (with amount and currency), stock, manufacturer_details (with name, address, and country), ratings (with rating_date, rating, and optional comments), benefits, optional precautions, optional related_products, availability (with online and in_store), optional expiration_date, and optional notes. Ensure variety in the data and realistic values.

  ${parser.getFormatInstructions()}`;

  console.log("Generating synthetic data...");

  const response = await llm.invoke(prompt);
  return parser.parse(response.content as string);
}

async function createProductSummary(product: Product): Promise<string> {
  return new Promise((resolve) => {
    const productDetails = `${product.product_name} (${product.category}): ${product.description}`;
    const ingredients = `Ingredients: ${product.ingredients.join(", ")}`;
    const usageInstructions = `Usage: ${product.usage_instructions}`;
    const price = `Price: ${product.price.amount} ${product.price.currency}`;
    const manufacturerDetails = `Manufactured by ${product.manufacturer_details.name}, ${product.manufacturer_details.address}, ${product.manufacturer_details.country}`;
    const stock = `Stock Available: ${product.stock}`;
    const ratings = product.ratings
      .map(
        (rating) =>
          `Rated ${rating.rating} on ${rating.rating_date}${
            rating.comments ? `: ${rating.comments}` : ""
          }`
      )
      .join(" ");
    const benefits = `Benefits: ${product.benefits.join(", ")}`;
    const precautions = product.precautions
      ? `Precautions: ${product.precautions}`
      : "";
    const relatedProducts = product.related_products
      ? `Related Products: ${product.related_products.join(", ")}`
      : "";
    const availability = `Available online: ${product.availability.online}, in-store: ${product.availability.in_store}`;
    const expirationDate = product.expiration_date
      ? `Expires on: ${product.expiration_date}`
      : "";
    const notes = product.notes ? `Notes: ${product.notes}` : "";

    const summary = `${productDetails}. ${ingredients}. ${usageInstructions}. ${price}. ${manufacturerDetails}. ${stock}. Reviews: ${ratings}. ${benefits}. ${precautions} ${relatedProducts}. ${availability}. ${expirationDate}. ${notes}`;

    resolve(summary);
  });
}

async function seedDatabase(): Promise<void> {
  try {
    await client.connect();
    await client.db("admin").command({ ping: 1 });
    console.log(
      "Pinged your deployment. You successfully connected to MongoDB!"
    );

    const db = client.db("prod_data");
    const collection = db.collection("products");

    await collection.deleteMany({});

    const syntheticData = await generateSyntheticData();

    const recordsWithSummaries = await Promise.all(
      syntheticData.map(async (record) => ({
        pageContent: await createProductSummary(record),
        metadata: { ...record },
      }))
    );

    for (const record of recordsWithSummaries) {
      await MongoDBAtlasVectorSearch.fromDocuments(
        [record],
        new OpenAIEmbeddings(),
        {
          collection,
          indexName: "vector_index",
          textKey: "embedding_text",
          embeddingKey: "embedding",
        }
      );

      console.log(
        "Successfully processed & saved record:",
        record.metadata.product_id
      );
    }

    console.log("Database seeding completed");
  } catch (error) {
    console.error("Error seeding database:", error);
  } finally {
    await client.close();
  }
}

seedDatabase().catch(console.error);
