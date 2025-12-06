import express from "express";
import dotenv from "dotenv";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { OpenAIEmbeddings } from "@langchain/openai";
import { FaissStore } from "@langchain/community/vectorstores/faiss";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(express.json());

// ----------- VECTOR STORE INIT -----------
const VECTOR_DIR = path.join(__dirname, "vectors");

let vectorStore = null;

async function loadVectorStore() {
  try {
    console.log("Loading FAISS index...");
    vectorStore = await FaissStore.load(
      VECTOR_DIR,
      new OpenAIEmbeddings({
        apiKey: process.env.OPENAI_API_KEY,
      })
    );
    console.log("FAISS loaded.");
  } catch (error) {
    console.log("No existing FAISS index found. Creating a new one...");
    vectorStore = await FaissStore.fromTexts(
      ["initial document"],
      [{ id: 1 }],
      new OpenAIEmbeddings({ apiKey: process.env.OPENAI_API_KEY })
    );
    await vectorStore.save(VECTOR_DIR);
    console.log("Created new FAISS index.");
  }
}

await loadVectorStore();

// ----------- EMBEDDING ENDPOINT -----------
app.post("/embed", async (req, res) => {
  try {
    const { text } = req.body;
   if (!text) return res.status(400).json({ error: "Missing 'text' parameter" });

