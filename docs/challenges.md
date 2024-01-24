## Nothing but a catalogue

Ok, you have decided that you need unstructured data search in your company, but the way before you'll provide production ready service, you need to show some Demo to your manager or even CEO, meanwhile you have nothing but a `catalogue` - a JSON or a CSV file with some items.

Use Embedding Studio default container to:

1. Fine-tune an embedding with your catalogue.
2. Quickly get a local search engine to show all the power of real unstructured search. 

## Static search quality

You have integrated an unstructured search into your platform or intra-net, most probably you did the following:

1. Chose an embedding related to your data.
2. Chose and integrated some Vector DB.

But you have faced the following problem - **your user experience is not improving**. 
Users actively search, click, visit customer's card, but on the same or nearly the same search gets the same results constantly. 

That's exactly for what the Embedding Studio was implemented for. 


## User experience improvement takes too long

Let's assume that you've implemented model improvement loop - you collect users clickstream and fine-tune your embedding model with it. 
But each fine-tuning cycle takes at least a day, you'll probably will not run fine-tuning each hour, because it's very expensive. 

Embedding Studio team is implementing **nearly online algorithm of Vector DB updating**, click the [`Watch` button](https://github.com/EulerSearch/embedding_studio) or write your request/questing to [our team](mailto:alexander@yudaev.ru).

## Slow and resource exhausted index updating

Your embedding model is huge, or your data domain items are huge, it doesn't matter, because you need lots of resources and time to fine-tune your base model.

We take this situation in count, too. Lightweight and fast embedding adapters which use only the vector data to be tuned are going to be released soon, click the [`Watch` button](https://github.com/EulerSearch/embedding_studio) to not miss it.


## Mix of structured and unstructured search

It's a highly rare case when a company will use unstructured search as is. And by searching `brick red houses san francisco area for april` user definitely wants to find some houses in San Francisco for a month-long rent in April, and then maybe brick-red houses. Unfortunately, for the 15th January 2024 there is no such accurate embedding model. So, companies need to mix structured and unstructured search.

The very first step of mixing it - to parse a search query. Usual approaches are:

1. Implement a bunch of rules, regexps, or grammar parsers (like NLTK grammar parser).
2. Collect search queries and to annotate some dataset for NER task. 

It takes some time to do, but at the end you can get controllable and very accurate query parser.

Embedding Studio team decided to dive into [LLM instruct fine-tuning]() for Zero-Shot query parsing task to close the first gap while a company doesn't have any rules and data being collected, or even eliminate exhausted rules implementation, but in the future.

## Structured search with unstructured queries

You don't need any unstructured search underneath you search system, that's ok, especially if you're a service like taxi or booking, because your customers usually want to arrive exactly to `221B Baker Street, London`, and nothing `very similar`.  As mentioned [in the previous paragraph](#mix-of-structured-and-unstructured-search) you'll probably try to implement search queries parser.

And so here we are, Embedding Studio can help you too in two ways:

1. Our [Zero-Shot Search queries parser](https://huggingface.co/EmbeddingStudio/query-parser-falcon-7b-instruct), which doesn't need anything except filters schema.
2. With capability of Vector DB and Embedding Studio you can implement query to address mapper, which will improve session by session.  

## Fresh items are getting lost

You upload new items periodically, and they get lost in new searches. This is not a problem, Embedding Studio has this feature in a backlog too, just click the [`Watch` button](https://github.com/EulerSearch/embedding_studio) not to miss it.

