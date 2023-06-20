<div align="center">

<h1></h1>

<img src="https://moe-counter.glitch.me/get/@LTMC?theme=rule34" /><br>

</div>

------

# long_term_memory_chat
This project leverages code from the [hyperDB](https://github.com/jdagdelen/hyperDB) repository, a powerful database

Welcome to Long Term Memory Chat Assistant, an innovative conversational solution that leverages cutting-edge AI models to deliver a dynamic, engaging, and personalized user experience.

The unique feature of this chat assistant lies in its ability to store and reference past conversations, thus providing a context-aware dialogue and enhancing the depth and relevance of its responses. This "long-term memory" functionality represents a significant shift from traditional chatbots, which often struggle to maintain conversational context beyond a few immediate exchanges.

Under the hood, this chat assistant is powered by OpenAI's GPT-3.5 model, renowned for its human-like text generation capability. It's also integrated with Sentence Transformers' 'all-MiniLM-L6-v2' model, a high-performance solution designed for computing semantically meaningful sentence embeddings. This ensures a high level of understanding and semantic accuracy in responses, enabling the assistant to handle a wide range of conversational topics effectively.

With its capacity to remember and learn from previous interactions, Long Term Memory Chat Assistant opens up exciting new possibilities for conversational AI, making interactions more fluid, personalized, and engaging than ever before. It's an excellent tool for anyone looking to explore the potential of AI-powered conversation with an emphasis on long-term memory and context-awareness.

Whether you're a developer, a researcher, or simply an AI enthusiast, we invite you to experience the advanced capabilities of Long Term Memory Chat Assistant. Get started with the setup instructions provided in this repository and start your journey into the future of conversational AI. Enjoy the experience!

## Features
- Long-term memory for retaining conversation history
- Ability to query previous conversations for context awareness and richer responses
- Multiple personas for varied interaction styles

  
# How it works
The core functionality of the chat assistant is in the handle_message function(main.py). 
Here is a brief rundown of its operation:
1. The function retrieves the current conversation history.
2. It appends the user's message to the conversation history.
3. The database is queried with the user's message and any results with similarity scores over a certain threshold (0.55 in this example) are filtered out.
4. These highly similar previous conversations, if any, are added to the user's message as 'Previous Conversation'.
5. The entire conversation history, including the user's latest message and any relevant 'Previous Conversations', is sent to the GPT-3.5 model.
6. The model generates a response which is added to the conversation history.
7. This updated conversation history is stored in the database and the database is saved to disk.
8. Finally, the assistant's response is sent back to the user.

Here's an example of how this process works:
![alt text](images/telegram.png)

When the user prompts "How are you today?", the assistant queries the database with the prompt, and the similar past conversations are added to the prompt as 'Previous Conversation':

![alt text](images/terminal.png)

This augmented prompt is then sent to the GPT-3.5 model for response generation.
![alt text](images/terminal_augmented_prompt.png)

# Setup & Installation
## Prerequisites
- Python 3.7+
- pip (Python package installer)

  
## Steps
1. Clone the repository:
```
git clone https://github.com/CCRss/long_term_memory_chat.git
```
2. Change the working directory to long_term_memory_chat:
```
cd long_term_memory_chat
```
3. Install the required packages using pip:
```
pip install -r requirements.txt
```
4. Update the `api_keys.json` file in the project directory with your OpenAI key and Telegram key:

```
{
"openai_key": "Your OpenAI key here", 
"telegram_key": "Your Telegram key here"
}
```


# Usage
To start the chat assistant, run the following command:
```
python main.py
```
Once the chat assistant is running, you can choose different bot personalities using the /start command in your chat interface. This command allows you to select from a range of predefined personality profiles for the assistant, enhancing the variability and depth of your interactions.



# Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


# License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for more information.
