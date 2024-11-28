import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { MongoClient } from "mongodb";
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb";
import xlsx from "xlsx";
import { z } from "zod";
import "dotenv/config";

const client = new MongoClient(process.env.MONGODB_ATLAS_URI as string);

const HealthCareProductSchema = z.object({
  product_id: z.string(),
  product_brand: z.string(),
  category: z.string(),
  product_name: z.string(),
  description: z.string(),
  benefits: z.array(z.string()),
  ingredients: z.array(z.string()),
  beauty_needs: z.string().optional(),
  beauty_justification: z.string().optional(),
  hair_skin_type: z.string().optional(),
  product_use: z.string(),
  product_application: z.string(),
});

type Product = z.infer<typeof HealthCareProductSchema>;

function readExcelFile(filePath: string): Product[] {
  const workbook = xlsx.readFile(filePath);
  const sheetName = workbook.SheetNames[0];
  const sheet = workbook.Sheets[sheetName];
  const rows = xlsx.utils.sheet_to_json(sheet);

  return rows.map((row: any, index: number) => ({
    product_id: (row.GTIN || `product_${index}`).toString(),
    product_brand: row.Brand || "Unknown Brand",
    category: row["Marketing range"] || "General",
    product_name: row["Commercial name"] || "Unknown Product",
    description: row["Product description"] || "No description provided",
    benefits: (row["Benefits"] || "").split(", "),
    ingredients: (row["Ingredients list"] || "").split(", "),
    beauty_needs: row["Beauty needs"] || "",
    beauty_justification: row["Beauty justification"] || "",
    hair_skin_type: row["Hair/skin Types"] || "",
    product_use: row["Frequency of use"] || "",
    product_application: row.Application || "",
  }));
}

async function createProductSummary(product: Product): Promise<string> {
  const productDetails = `${product.product_name} (${product.category}): ${product.description}`;
  const ingredients = `Ingredients: ${product.ingredients.join(", ")}`;
  const benefits = `Benefits: ${product.benefits.join(", ")}`;
  const productUse = `Use: ${product.product_use}`;
  const application = `Application: ${product.product_application}`;

  return `${productDetails}. ${ingredients}. ${benefits}. ${productUse}. ${application}`;
}

async function seedDatabase(): Promise<void> {
  try {
    await client.connect();
    await client.db("admin").command({ ping: 1 });
    console.log("Pinged your deployment. You successfully connected to MongoDB!");

    const db = client.db("prod_data");
    const collection = db.collection("products");

    await collection.deleteMany({});

    const syntheticData = readExcelFile("AveneData.xlsx");

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

