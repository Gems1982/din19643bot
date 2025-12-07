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

// ---------------- VECTOR STORE ----------------
const VECTOR_DIR = path.join(__dirname, "vectors");
let vectorStore = null;

async function initVectorStore() {
  console.log("🔍 Checking FAISS vector index...");

  try {
    // Try loading existing FAISS index
    vectorStore = await FaissStore.load(
      VECTOR_DIR,
      new OpenAIEmbeddings({
        apiKey: process.env.OPENAI_API_KEY,
      })
    );
    console.log("✅ Loaded existing FAISS index.");
  } catch (err) {
    console.log("⚠️ No FAISS index found — creating new one...");

    vectorStore = await FaissStore.fromTexts(
      ["empty index"],
      [{ id: 0 }],
      new OpenAIEmbeddings({
        apiKey: process.env.OPENAI_API_KEY,
      })
    );

    await vectorStore.save(VECTOR_DIR);
    console.log("✅ New FAISS index created.");
  }
}

// ---------------- API ENDPOINTS ----------------

// Add text to FAISS
app.post("/embed", async (req, res) => {
  try {
    const { text, metadata } = req.body;

    if (!text) {
      return res.status(400).json({ error: "Missing 'text' field" });
    }

    console.log("🔹 Embedding text:", text.substring(0, 50));

    await vectorStore.addTexts([text], [metadata || {}]);

    await vectorStore.save(VECTOR_DIR);

    res.json({ success: true });
  } catch (err) {
    console.error("❌ Error embedding text:", err);
    res.status(500).json({ error: "Embedding failed", details: err.message });
  }
});

// Query FAISS
app.post("/query", async (req, res) => {
  try {
    const { query } = req.body;

    if (!query) {
      return res.status(400).json({ error: "Missing 'query' field" });
    }

    console.log("🔍 Searching for:", query);

    const results = await vectorStore.similaritySearch(query, 5);

    res.json({ results });
  } catch (err) {
    console.error("❌ Query error:", err);
    res.status(500).json({ error: "Query failed", details: err.message });
  }
});

// ---------------- START SERVER ----------------
async function start() {
  await initVectorStore();

  const PORT = process.env.PORT || 3000;
  app.listen(PORT, () => {
    console.log(`🚀 Server running on port ${PORT}`);
  });
}

start();
