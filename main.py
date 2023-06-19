from telegram import Update, ForceReply 
from telegram.ext import Updater, CommandHandler, MessageHandler,  CallbackContext, Filters, CallbackQueryHandler
import openai
import json
from hyperdb import init_db
from hyperdb.telegram_buttons import start, button

db, db_file = init_db() # Initialize database


# Get API-key
def get_api_key(file_name: str, key_name: str) -> str:
    with open(file_name, 'r') as f:
        config = json.load(f)
    return config[key_name]

openai.api_key = get_api_key("api_keys.json", "openai_key") # Set API-key


# Initialize the conversation history
conversation_history = [
    {"role": "system", "content": "Say 'meow' at the end of each sentence because you're a catgirl. You should remember previous messages if there is a 'Previous Conversation' message."},
]


def handle_message(update: Update, context: CallbackContext) -> None:
    personality = context.user_data.get('personality', 'default') # Use 'default' if no personality has been chosen yet

    # Get the conversation history for the chosen personality
    conversation_histories = context.user_data.get('conversation_history', {})
    conversation_history = conversation_histories.get(personality, [])

    message = update.effective_message
    conversation_history.append({"role": "user", "content": message.text}) # Add the user's message to the conversation history

    query_results = db.query(message.text) # Query the database with the user's message
   
    filtered_results = [result for result in query_results if result[1] > 0.63]  # Filter the results to only include documents with similarity > 0.63

    # If there are any filtered results, add them to the user's message
    if filtered_results:
        for result in filtered_results:
            message.text += f"\n\nPrevious Conversation:\n{result[0]}"  # Get the message of each similar result

    conversation_history.append({"role": "system", "content":  message.text})

    print("|||||||||||||||||||||||")
    print(f"Prompt: {message.text}")
    print("|||||||||||||||||||||||")
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation_history  # Use the conversation history
    )
    
    assistant_reply = response['choices'][0]['message']['content'] # Get the assistant's reply from the response

    conversation_history.append({"role": "assistant", "content": assistant_reply}) # Add the assistant's reply to the conversation history

    # Save the conversation to the database
    document = {
        "message": f"\nUser: {message.text}\nAssistant: {assistant_reply}",
        "role": "conversation"
    }

    db.add_documents([document])

    db.save(db_file) # Save the updated database to the file

    message.reply_text(assistant_reply) # Send the assistant's reply back to the user

    # Print the query results
    for document, similarity in query_results:
        if similarity >= 0.30:
            print(f"Document: {document}\nSimilarity: {similarity}\n")
    
    context.user_data['conversation_history'] = conversation_history # Update the user's conversation history

    # At the end of the function, save the updated conversation history
    conversation_histories[personality] = conversation_history
    context.user_data['conversation_history'] = conversation_histories


    

def main() -> None:
    # Set up the Updater
    updater = Updater(token=get_api_key("api_keys.json", "telegram_key"), use_context=True)

    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CallbackQueryHandler(button))
    
    dispatcher.add_handler(MessageHandler(Filters.text, handle_message)) # Add a message handler that processes all text messages

    updater.start_polling() # Start the Bot

    updater.idle() # Run the bot until you press Ctrl-C


if __name__ == '__main__':
    main()