import { HfInference } from "@huggingface/inference";
import dotenv from "dotenv";
dotenv.config();

const hf = new HfInference(process.env.HUGGINGFACE_API_KEY);

function cosineSimilarity(vecA: number[], vecB: number[]): number {
  if (vecA.length !== vecB.length) {
    throw new Error("Vectors must have the same length");
  }
  const dotProduct = vecA.reduce((sum, a, i) => sum + a * (vecB[i] ?? 0), 0);
  const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
  const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
  if (magnitudeA === 0 || magnitudeB === 0) return 0;
  return dotProduct / (magnitudeA * magnitudeB);
}

async function getEmbedding(text: string): Promise<number[]> {
  const response = await hf.featureExtraction({
    model: "sentence-transformers/all-MiniLM-L6-v2",
    inputs: text,
  });

  if (Array.isArray(response) && Array.isArray(response[0])) {
    return response[0] as number[];
  }
  return response as number[];
}

async function main() {
  console.log("Generating embeddings and comparing similarity...\n");

  const sentences = [
    "Delhi has severe air pollution in winter",
    "Air quality in Delhi worsens during November and December",
    "Machine learning models require large amounts of training data"
  ];

  const embeddings = await Promise.all(
    sentences.map(s => getEmbedding(s))
  );

  console.log(`Embedding size: ${embeddings[0]!.length} numbers per sentence\n`);

  const sim12 = cosineSimilarity(embeddings[0]!, embeddings[1]!);
  const sim13 = cosineSimilarity(embeddings[0]!, embeddings[2]!);
  const sim23 = cosineSimilarity(embeddings[1]!, embeddings[2]!);

  console.log("Similarity scores (closer to 1 = more similar meaning):\n");
  console.log(`"Delhi pollution" vs "Delhi air quality":  ${sim12.toFixed(4)}`);
  console.log(`"Delhi pollution" vs "Machine learning":   ${sim13.toFixed(4)}`);
  console.log(`"Delhi air quality" vs "Machine learning": ${sim23.toFixed(4)}`);

  console.log("\nNote: sentences 1 and 2 share almost no words");
  console.log("but should score much higher similarity than sentence 3.");
  console.log("That's embeddings working — meaning captured mathematically.");
}

main();