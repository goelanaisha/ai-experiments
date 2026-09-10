import { HfInference } from "@huggingface/inference";
import Groq from "groq-sdk";
import * as fs from "fs";
import * as path from "path";
import dotenv from "dotenv";
dotenv.config();

const hf = new HfInference(process.env.HUGGINGFACE_API_KEY);
const groq = new Groq({apiKey: process.env.GROQ_API_KEY});

interface chunk {
    id:number;
    text:string;
    embedding:number[];
    source:string;
}
const VectorStore :chunk[]= [];

async function getEmbeddings(text:string):Promise<number[]> {
    const response = await hf.featureExtraction({
        model: "sentence-transformers/all-MiniLM-L6-v2",
        inputs: text,
    });
    if (Array.isArray(response) && Array.isArray(response[0])){
        return response[0] as number[];
    }
    return response as number[];
}
function cosineSimilarity(vecA:number[], vecB:number[]):number {
     const dotProduct = vecA.reduce((sum, a, i) => sum + a * (vecB[i] ?? 0), 0);
    const magnitudeA = Math.sqrt(vecA.reduce((sum,a)=> sum + a * a, 0));
    const magnitudeB = Math.sqrt(vecB.reduce((sum,b)=> sum + b * b, 0));
    if(magnitudeA==0 || magnitudeB==0) return 0;
    return dotProduct / (magnitudeA * magnitudeB);
}
function chunkText(text:string,chunkSize:number=200,overlap:number=50):string[]{
    const words = text.split(/\s+/);
    const chunks:string[] = [];
     
   let i =0;
   while(i<words.length){
        const chunk = words.slice(i,i+chunkSize).join(" ");
        chunks.push(chunk);
        i += chunkSize - overlap;
   }
   return chunks;
}
async function indexFile(filePath:string):Promise<void>{
    console.log(` \nIndexing : ${path.basename(filePath)}`);

    const text = fs.readFileSync(filePath, "utf-8");
    const chunks = chunkText(text);

    console.log( `  split into ${chunks.length} chunks`);

    for(let i=0;i< chunks.length;i++){
        const embedding = await getEmbeddings(chunks[i]!);
        VectorStore.push({
            id: VectorStore.length,
            text: chunks[i]!,
            embedding,
            source: path.basename(filePath)
        });
        process.stdout.write(` embedding chunk ${i+1}/${chunks.length}\r`);
    }
    console.log(`  ✓ Done — ${chunks.length} chunks indexed`);

}
function retrieve(queryEmbedding: number[], topK: number): (chunk & { similarity: number })[] {
  return VectorStore
    .map(chunk => ({
      ...chunk,
      similarity: cosineSimilarity(queryEmbedding, chunk.embedding)
    }))
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, topK);
}
async function ragQuery(question: string): Promise<void> {
  console.log(`\nQuestion: "${question}"`);
  console.log("─".repeat(50));

  const questionEmbedding = await getEmbeddings(question);
  const relevantChunks = retrieve(questionEmbedding, 3);

  console.log("\nRetrieved chunks:");
  relevantChunks.forEach((chunk, i) => {
    console.log(`  [${i + 1}] (${chunk.similarity.toFixed(4)}) from ${chunk.source}`);
    console.log(`       "${chunk.text.substring(0, 80)}..."`);
  });
  const context = relevantChunks
    .map((c, i) => `[Source ${i + 1} - ${c.source}]\n${c.text}`)
    .join("\n\n");

  const stream = await groq.chat.completions.create({
    model: "openai/gpt-oss-20b",
    max_tokens: 512,
    stream: true,
    messages: [
      {
        role: "system",
        content: `You are a helpful assistant. Answer using ONLY the context below.
Always mention which source your answer comes from.
If the context doesn't contain the answer, say "I don't have enough information."

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
  // Create a sample knowledge file if it doesn't exist
  const sampleFile = "delhi-aqi-knowledge.txt";

  if (!fs.existsSync(sampleFile)) {
    fs.writeFileSync(sampleFile, `
Delhi Air Quality Guide

Delhi experiences some of the worst air quality in the world, particularly during winter months from October to January. The Air Quality Index (AQI) regularly crosses 400 during this period, which falls in the Severe category.

Major Sources of Pollution
The primary sources of air pollution in Delhi include vehicular emissions from over 11 million registered vehicles, industrial emissions from factories in and around Delhi, crop stubble burning in neighboring states Punjab and Haryana during October and November, construction dust from rapid urbanization, and seasonal factors like temperature inversions that trap pollutants close to the ground.

Health Impact
Exposure to high AQI levels causes both short-term and long-term health effects. Short-term effects include eye irritation, throat irritation, coughing, and difficulty breathing. Long-term exposure to PM2.5 particles smaller than 2.5 micrometers can penetrate deep into the lungs and cause chronic respiratory diseases, cardiovascular problems, and has been linked to lung cancer.

Protective Measures
When AQI exceeds 150, residents should wear N95 or N99 masks outdoors as they filter 95-99% of harmful particles. Surgical masks and cloth masks offer minimal protection against PM2.5 particles. Indoor air purifiers with HEPA filters can remove up to 99.97% of particles and significantly improve indoor air quality. Keeping windows closed during high pollution periods helps prevent outdoor pollutants from entering homes.

Government Response
The Graded Response Action Plan (GRAP) is implemented by the Commission for Air Quality Management when AQI crosses certain thresholds. At AQI above 200, GRAP Stage 1 restricts garbage burning. At AQI above 300, Stage 2 bans diesel generators. At AQI above 400, Stage 3 stops construction activities. At AQI above 450, Stage 4 considers odd-even vehicle rationing and school closures.

AQI Scale
The AQI scale in India ranges from 0 to 500. 0-50 is Good and poses no health risk. 51-100 is Satisfactory with minor breathing discomfort for sensitive people. 101-200 is Moderate causing breathing discomfort during prolonged outdoor activity. 201-300 is Poor causing breathing discomfort for most people. 301-400 is Very Poor causing respiratory illness on prolonged exposure. 401-500 is Severe affecting healthy people and seriously impacting those with existing diseases.
    `.trim());
    console.log("Created sample file: delhi-aqi-knowledge.txt\n");
  }

  // Index the file
  await indexFile(sampleFile);

  console.log(`\nVector store ready — ${VectorStore.length} chunks indexed`);
  console.log("=".repeat(50));

  // Ask real questions about the file
  await ragQuery("What mask should I wear when pollution is high?");
  await ragQuery("What happens under GRAP Stage 3?");
  await ragQuery("How does long term pollution exposure affect health?");
  await ragQuery("What is the AQI level considered severe?");
}

main();