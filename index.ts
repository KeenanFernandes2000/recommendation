import "dotenv/config";
import express, { Express, Request, Response } from "express";
import { MongoClient } from "mongodb";
import { callAgent } from "./agent";
import multer from "multer";

const app: Express = express();


app.use(express.json());

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, "public/images/");
  },
  filename: function (req, file, cb) {
    cb(null, file.originalname); // specify the filename for the uploaded file
  },
});
const upload = multer({ storage: storage });

// Initialize MongoDB client
const client = new MongoClient(process.env.MONGODB_ATLAS_URI as string);

async function startServer() {
  try {
    await client.connect();
    await client.db("admin").command({ ping: 1 });
    console.log(
      "Pinged your deployment. You successfully connected to MongoDB!"
    );

    // Set up basic Express route
    // curl -X GET http://localhost:3000/
    app.get("/", (req: Request, res: Response) => {
      res.send("LangGraph Agent Server");
    });

    // API endpoint to start a new conversation
    // curl -X POST -H "Content-Type: application/json" -d '{"message": "Build a team to make an iOS app, and tell me the talent gaps."}' http://localhost:3000/chat
    app.post("/chat", upload.single("image"), async (req: Request, res: Response) => {
      const {
        skinType,
        mainConcerns,
        breakoutFrequency,
        productSensitivity,
        skinCareRoutine
      } = req.body;

      const image = req.file;
      const path = image?.path;
      const threadId = Date.now().toString();

      // Combine questionnaire responses into a structured format
      const userResponses = {
        skinType,
        mainConcerns,
        breakoutFrequency,
        productSensitivity,
        skinCareRoutine
      };

      try {
        const response = await callAgent(
          client,
          JSON.stringify(userResponses),
          threadId,
          path || null
        );
        res.json({ threadId, response });
      } catch (error) {
        console.error("Error starting conversation:", error);
        res.status(500).json({ error: "Internal server error" });
      }
    });

    // API endpoint to send a message in an existing conversation
    // curl -X POST -H "Content-Type: application/json" -d '{"message": "What team members did you recommend?"}' http://localhost:3000/chat/123456789
    app.post("/chat/:threadId", async (req: Request, res: Response) => {
      const { threadId } = req.params;
      const { message } = req.body;
      try {
        const response = await callAgent(client, message, threadId, null);
        res.json({ response });
      } catch (error) {
        console.error("Error in chat:", error);
        res.status(500).json({ error: "Internal server error" });
      }
    });

    const PORT = process.env.PORT || 3000;
    app.listen(PORT, () => {
      console.log(`Server running on port ${PORT}`);
    });
  } catch (error) {
    console.error("Error connecting to MongoDB:", error);
    process.exit(1);
  }
}

startServer();
