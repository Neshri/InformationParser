# Assumptions:
# - The task is a single string
# - The conversation is a list of dictionaries, where each dictionary has a 'message', 'role', and 'processed' key
# - Available roles are, 'user', 'ai', 'system', and 'tool'

import ollama
import numpy as np
import util_model

IMPORTANT_NUMBERS = {
    "score_threshold": 0.7
}

def get_overarching_context_embedding(task, conversation, max_tokens=4096):
    """
    Generate an overarching context embedding by combining the task with recent context.
    """
    # Start with the task
    context = task + "\n"
    
    # Append recent user and AI messages until reaching max tokens
    for entry in reversed(conversation):
        if 'processed' in entry.keys() and entry['processed'] == True:
            context = entry['message'] + "\n" + context
        if len(context) > max_tokens:
            context = context[:max_tokens]
            break
    
    # Generate and return the embedding for the combined context
    return get_embeddings(context)


def get_embeddings(text):
    # Return the vector/embedding for the given text
    return ollama.embed(model='nomic-embed-text', input=text).embeddings[0]

def get_embedding_similarity(embedding_a, embedding_b):
    similarity = np.dot(embedding_a, embedding_b) / (np.linalg.norm(embedding_a) * np.linalg.norm(embedding_b))
    return similarity

def get_text_similarity(text_a, text_b):
    if isinstance(text_b, list):
        embeddings = [np.array(get_embeddings(t)) for t in text_b]
        similarities = [np.dot(np.array(get_embeddings(text_a)), e) / (np.linalg.norm(np.array(get_embeddings(text_a))) * np.linalg.norm(e)) for e in embeddings]
        return np.mean(similarities)
    else:
        embedding_a = np.array(get_embeddings(text_a))
        embedding_b = np.array(get_embeddings(text_b))
        similarity = np.dot(embedding_a, embedding_b) / (np.linalg.norm(embedding_a) * np.linalg.norm(embedding_b))
        return similarity

def clean_entry(entry, conversation, original_task):
    new_entry = {}
    overarch_context = get_overarching_context_embedding(original_task, conversation)
    for key in entry.keys():
        if key == 'message':
            new_message = ""
            # Refactor the message
            if entry['role'] == 'user':
                new_message = util_model.query_clean_message(entry['message'])
            elif entry['role'] == 'ai':
                new_message = entry['message']
            elif entry['role'] == 'system':
                new_message = entry['message']
            elif entry['role'] == 'tool':
                chunks = entry['message'].split("\n\n")
                new_message = ""
                for chunk in chunks:
                    score = get_embedding_similarity(get_embeddings(chunk), overarch_context)
                    if score > IMPORTANT_NUMBERS['score_threshold']:
                        new_message += chunk
                if new_message == "":
                    new_message = "No relevant information found."
                pass
            else:
                pass
            new_entry[key] = new_message
        else:
            new_entry[key] = entry[key]

    new_entry['processed'] = True
    return new_entry

def clean_conversation(conversation):
    # Find original task
    for i, entry in enumerate(conversation):
        if entry['role'] == 'user':
            if ('processed' not in entry.keys() or entry['processed'] == False):
                clean_task = util_model.query_clean_message(entry['message'])
                conversation[i]['message'] = clean_task
                conversation[i]['processed'] = True
            original_task = entry['message']
            break

    # Find the index of the first unprocessed message
    new_i = 0
    for i in reversed(range(len(conversation))):
        if 'processed' in conversation[i].keys() and conversation[i]['processed'] == True:
            break
        new_i = i
    
    for i in range(new_i, len(conversation)):
        conversation[i] = clean_entry(conversation[i], conversation, original_task)
    return conversation

def print_conversation(conversation):
    for entry in conversation:
        print(f"{entry['role']}: {entry['message']}")

if __name__ == "__main__":
    # print(get_embeddings("Hello, world!"))
    # print(get_text_similarity("Hello, world!", "Goodbye, world!"))
    # print(get_text_similarity("Hello, Bob!", "Hello, world!"))
    # print(get_text_similarity("Hello, Bob!", "Goodbye, world!"))
    # print(get_text_similarity("Hello, Bob!", ["Hello, world!", "Goodbye, world!"]))

    conversation = [
    {"role": "user", "message": "I want to create an e-commerce platform that supports multiple payment gateways, product recommendations, and user reviews. I also need to track shipping status and handle customer support tickets."},
    {"role": "ai", "message": "That’s a big project! I suggest breaking it into modules. You can start with user authentication and a product catalog. Then, move on to the payment gateways and other features."},
    {"role": "system", "message": "Initializing e-commerce project setup with features: user authentication, product catalog, payment integration, and support ticket system."},
    {"role": "tool", "message": "Found some references on payment gateway APIs, including Stripe, PayPal, and others. They offer detailed documentation for integration."},
    {"role": "ai", "message": "For the product catalog, you’ll need a database to store product details, prices, and inventory. Should we discuss database structure next?"}, 
    {"role": "user", "message": "I think the payment gateways are the most urgent. Can you help me integrate PayPal?"}, 
    {"role": "tool", "message": "Here’s some code snippets and documentation on integrating PayPal into an e-commerce platform. You’ll need to set up an API key and configure webhooks for payment notifications."},
    {"role": "ai", "message": "To integrate PayPal, start by setting up the PayPal developer account and obtaining the credentials. Then, use the PayPal SDK for seamless transactions. I can help with the code."},
    {"role": "user", "message": "I also need a way to handle product recommendations based on user behavior. Can you suggest how to implement that?"}, 
    {"role": "tool", "message": "Here’s some information on recommendation algorithms, including collaborative filtering and content-based filtering. You can implement these using machine learning libraries."},
    {"role": "ai", "message": "For recommendations, you could use a collaborative filtering algorithm where users are recommended products based on their browsing or purchase history."},
    {"role": "user", "message": "I would also like to implement a real-time chat feature for customer support tickets. Any suggestions?"}, 
    {"role": "tool", "message": "I found documentation on integrating real-time chat APIs. Here are the features available, including message queues and user management for chat support."},
    {"role": "system", "message": "Tracking progress on modules: Payment Gateway - PayPal integration in progress, Product Recommendations - research phase, Customer Support - real-time chat setup."},
    {"role": "ai", "message": "To implement real-time chat, I suggest using WebSockets or Firebase for real-time communication. I can guide you on how to set this up."},
    {"role": "user", "message": "Let’s focus on shipping status next. How do I track shipments?"}, 
    {"role": "tool", "message": "Here’s some APIs that can be used to track shipments from various carriers, including FedEx and UPS. These provide real-time tracking and notifications."},
    {"role": "ai", "message": "You can integrate shipment tracking by using third-party APIs from carriers like FedEx, UPS, or DHL. You’ll need to handle API responses to update customers on the status of their orders."},
    {"role": "user", "message": "Can you suggest a way to handle the user reviews for products?"}, 
    {"role": "tool", "message": "I found some libraries for handling product reviews, including features like star ratings, reviews display, and moderation tools."},
    {"role": "ai", "message": "For user reviews, you can store them in the product database and use a moderation system to filter out inappropriate content. I can show you how to set that up."},
    {"role": "system", "message": "System check: Modules in progress - Payment Gateway, Product Recommendations, Customer Support, Shipment Tracking."},
    {"role": "tool", "message": "Found some more resources on AI-driven product recommendations based on deep learning models."},
    {"role": "ai", "message": "For advanced product recommendations, deep learning models like neural collaborative filtering could be used to improve recommendations over traditional methods."},
    {"role": "user", "message": "I think I have a good overview now. I’ll start with the payment integration and build from there."},
    {"role": "ai", "message": "Great! Once you’re done with the payment gateway integration, let me know and we’ll move on to the next module."}
]

    print_conversation(conversation)
    conversation = clean_conversation(conversation)
    print("Cleaned conversation:")
    print_conversation(conversation)
    
