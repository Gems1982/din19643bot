import fs from "fs-extra";
import path from "path";
import dotenv from "dotenv";
import { OpenAI } from "openai";
dotenv.config();

const KB_DIR = "./kb";
const VECTOR_DIR = "./vectors";
const META_FILE = `${VECTOR_DIR}/kb_meta.json`;
const VECTORS_FILE = `${VECTOR_DIR}/kb_vectors.json`;

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

function chunkText(text, maxTokens = 500) {
  const sentences = text.split(/(?<=[.!?])\s+/);
  let chunks = [];
  let current = "";

  for (const sentence of sentences) {
    if ((current + sentence).length > maxTokens * 4) {
      chunks.push(current.trim());
      current = "";
    }
    current += sentence + " ";
  }
  if (current.trim()) chunks.push(current.trim());
  return chunks;
}

async function embedKnowledgeBase() {
  console.log("📁 Loading KB files…");

  const files = await fs.readdir(KB_DIR);

  let chunks = [];
  let metadata = [];

  for (const file of files) {
    const text = await fs.readFile(path.join(KB_DIR, file), "utf-8");
    const fileChunks = chunkText(text);

    for (const chunk of fileChunks) {
      chunks.push(chunk);
      metadata.push({ file, text: chunk });
    }
  }

  console.log("🧠 Total chunks:", chunks.length);

  let vectors = [];

  for (let i = 0; i < chunks.length; i++) {
    console.log(`Embedding ${i + 1}/${chunks.length}`);

    const res = await client.embeddings.create({
      model: "text-embedding-3-large",
      input: chunks[i]
    });

    vectors.push(res.data[0].embedding);
  }

  await fs.ensureDir(VECTOR_DIR);

  await fs.writeJson(META_FILE, metadata, { spaces: 2 });
  await fs.writeJson(VECTORS_FILE, vectors, { spaces: 2 });

  console.log("🎉 Embedding complete!");
}

embedKnowledgeBase();
