from telegram import InlineKeyboardButton, InlineKeyboardMarkup
import json

def start(update, context):
    with open("personalities.json", 'r') as f:
        personalities = json.load(f)

    keyboard = [[InlineKeyboardButton(p, callback_data=p)] for p in personalities.keys()]

    reply_markup = InlineKeyboardMarkup(keyboard)

    update.message.reply_text('Please choose your assistant\'s personality:', reply_markup=reply_markup)

def button(update, context):
    query = update.callback_query

    query.answer()

    new_personality = query.data 

    # context.user_data['conversation_history'] is now a dict mapping personalities to their conversation histories
    conversation_histories = context.user_data.get('conversation_history', {})

    # If the new personality does not have a conversation history, we initialize a new one
    if new_personality not in conversation_histories:
        conversation_histories[new_personality] = [{"role": "system", "content": "You are " + new_personality}]

    context.user_data['personality'] = new_personality
    context.user_data['conversation_history'] = conversation_histories

    query.edit_message_text(text=f"Selected option: {new_personality}")
