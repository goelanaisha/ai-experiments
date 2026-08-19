import Groq from "groq-sdk";
import dotenv from "dotenv";
dotenv.config();

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

async function main() {
  console.log("Response:\n");

  const completion = await groq.chat.completions.create({
    model: "llama-3.3-70b-versatile",
    max_tokens: 1024,
    messages: [
      {
        role: "system",
        content: "Act as a tech mentor and help to understand and learn technology."
      },
      {
        role: "user",
        content: "Explain what an API is in 3 sentences."
      }
    ]
  });

  const content = completion.choices?.[0]?.message?.content ?? "";
  process.stdout.write(content);

  console.log("\n\n--- Stats ---");
  console.log(`Input tokens used: ${completion.usage?.prompt_tokens ?? 0}`);
  console.log(`Output tokens generated: ${completion.usage?.completion_tokens ?? 0}`);
}

main();