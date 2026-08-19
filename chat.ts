import Groq from "groq-sdk";
import dotenv from "dotenv";
import * as readline from "readline";
dotenv.config();

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

const conversationHistory:{ role: "user" | "assistant"; content: string}[]=[];

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

function askQuestion(prompt: string): Promise<string>{
    return new Promise((resolve)=> rl.question(prompt,resolve));
}

async function chat(userMessage: string): Promise<string> {
    conversationHistory.push({
        role:"user",
        content:userMessage
    });

    const stream = await groq.chat.completions.create({
        model: "llama-3.3-70b-versatile",
        max_tokens:1024,
        messages:[
            {
                role:"system",
                content:"You are a helpful assistant that provides concise and clear answers to user questions."
            },
            ...conversationHistory
        ],
        stream: true
    });
    process.stdout.write("Assistant: ");

    let fullResponse = "";

   for await (const chunk of stream) {
    const delta: string = chunk.choices[0]?.delta?.content ?? "";
    process.stdout.write(delta);
    fullResponse += delta;
  }

    
    console.log("\n");
    conversationHistory.push({
        role:"assistant",
        content:fullResponse
    });

    return fullResponse;
}
async function main(){
    console.log("Welcome to the interactive chat! Type 'exit' to quit.\n");
    while(true){
        const userInput = await askQuestion("You: ");
        if(userInput.toLowerCase() === "exit"){
            console.log(`\nConversation ended. Total messages: ${conversationHistory.length}`);
            rl.close();
            break;
        }
        await chat(userInput);
    }

}
main();