import { HfInference } from "@huggingface/inference";
import Groq from "groq-sdk";
import dotenv from "dotenv";
dotenv.config();

const hf = new HfInference(process.env.HUGGINGFACE_API_KEY);
const groq = new Groq({apiKey: process.env.GROQ_API_KEY});

interface Document {
    id : number;
    text : string;
    embedding : number[];
}
const VectorStore : Document[]=[];

async function getEmbedding(text:string):Promise<number[]>{
    const response = await hf.featureExtraction({
        model: "sentence-transformers/all-MiniLM-L6-v2",
        inputs: text,
    });
    if (Array.isArray(response) && Array.isArray(response[0])){
        return response[0] as number[];
    }
    return response as number[];
}
function cosineSimilarity(vecA: number[], vecB: number[]): number {
    const dotProduct = vecA.reduce((sum,a,i) => sum + a *(vecB[i]??0), 0);
    const magnitudeA = Math.sqrt(vecA.reduce((sum,a)=> sum + a*a, 0));
    const magnitudeB = Math.sqrt(vecB.reduce((sum,b)=> sum + b*b, 0));
    if(magnitudeA === 0 || magnitudeB === 0){
        return 0;
    }
    return dotProduct / (magnitudeA * magnitudeB);
}
async function addDocument(text: string): Promise<void> {
    const embedding = await getEmbedding(text);
    VectorStore.push({ id: VectorStore.length + 1, text, embedding });
    console.log(` ⎷ Stored: "${text.substring(0,50)}..."`);
}

function retrieve(queryEmbedding: number[], topK: number): Document[] {
  return VectorStore
    .map(doc => ({
      ...doc,
      similarity: cosineSimilarity(queryEmbedding, doc.embedding)
    }))
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, topK);
}

async function ragQuery(question: string): Promise<void> {
  console.log(`\nQuestion: "${question}"`);
  console.log("─".repeat(50));
const questionEmbedding = await getEmbedding(question);
const relevantDocs = retrieve(questionEmbedding, 2);
console.log("\nRetrieved context:");
  relevantDocs.forEach((doc, i) => {
    const sim = cosineSimilarity(questionEmbedding, doc.embedding);
    console.log(`  [${i + 1}] (similarity: ${sim.toFixed(4)}) ${doc.text}`);
  });
 const context = relevantDocs.map(d => d.text).join("\n");
 const stream = await groq.chat.completions.create({
    model: "openai/gpt-oss-20b",
    max_tokens: 512,
   stream: true,
    messages: [
      {
        role: "system",
        content: `You are a helpful assistant. Answer questions using ONLY the provided context. 
If the context doesn't contain enough information, say so clearly.
Context:
${context}`
      },
      {
        role: "user",
        content: question
      }
    ]
  });

  console.log("\nAnswer:");
  for await (const chunk of stream) {
    process.stdout.write(chunk.choices[0]?.delta?.content ?? "");
  }
  console.log("\n");
}
async function main() {
  // Your knowledge base — in a real app these would be chunks from PDFs/docs
  const documents = [
    "Delhi's AQI frequently exceeds 400 in November and December due to crop burning in neighboring states.",
    "The primary pollutants in Delhi air are PM2.5 and PM10 particles from vehicles and industry.",
    "An N95 mask filters 95% of airborne particles and is recommended when AQI exceeds 150.",
    "AQI between 0-50 is Good, 51-100 is Satisfactory, 101-200 is Moderate, 201-300 is Poor.",
    "The Graded Response Action Plan (GRAP) restricts construction and vehicle use during severe pollution.",
    "Indoor air purifiers with HEPA filters can reduce indoor pollution by up to 85%.",
    "Python is a popular programming language used extensively in data science and machine learning.",
    "React is a JavaScript library for building user interfaces developed by Meta.",
  ];
  console.log("Building vector store...\n");
  for (const doc of documents) {
    await addDocument(doc);
  }

  console.log(`\nVector store ready — ${VectorStore.length} documents indexed\n`);
  console.log("=".repeat(50));

  // Ask questions — watch which documents get retrieved
  await ragQuery("What should I wear outside when pollution is high?");
  await ragQuery("When does Delhi pollution get worst?");
  await ragQuery("How do I reduce pollution inside my home?");
}

main();