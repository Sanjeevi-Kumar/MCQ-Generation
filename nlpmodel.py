from textwrap3 import wrap
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize
import string
import pke
import traceback
import torch
from collections import OrderedDict
from transformers import T5ForConditionalGeneration, T5Tokenizer
from flashtext import KeywordProcessor
from sense2vec import Sense2Vec
import random


s2v = Sense2Vec().from_disk(r"C:\Users\Mugesh Ram Sundar\Desktop\Others\Sem 4 Project\mcq\nlp\s2v_old")

def sense2vec_get_words(word,s2v):
    output = []
    word = word.lower()
    word = word.replace(" ", "_")
    sense = s2v.get_best_sense(word)
    if sense is None:
          famous_names = [
            "Albert Einstein",
            "Isaac Newton",
            "Leonardo da Vinci",
            "Marie Curie",
            "William Shakespeare",
            "Charles Darwin",
            "Thomas Edison",
            "Nikola Tesla",
            "Ludwig van Beethoven",
            "Wolfgang Amadeus Mozart",
            "Johann Sebastian Bach",
            "Vincent van Gogh",
            "Pablo Picasso",
            "Michelangelo",
            "Aristotle",
            "Plato",
            "Socrates",
            "Alexander the Great",
            "Julius Caesar",
            "Cleopatra",
            "Queen Elizabeth I",
            "Winston Churchill",
            "Abraham Lincoln",
            "George Washington",
            "Thomas Jefferson",
            "Benjamin Franklin",
            "Mahatma Gandhi",
            "Nelson Mandela",
            "Martin Luther King Jr.",
            "Mother Teresa",
            "Christopher Columbus",
            "Ferdinand Magellan",
            "Marco Polo",
            "Neil Armstrong",
            "Yuri Gagarin",
            "Amelia Earhart",
            "Charles Lindbergh",
            "Ernest Hemingway",
            "Mark Twain",
            "Jane Austen",
            "Charles Dickens",
            "Fyodor Dostoevsky",
            "Leo Tolstoy",
            "James Joyce",
            "Virginia Woolf",
            "Gabriel Garcia Marquez",
            "J.K. Rowling",
            "George Orwell",
            "J.R.R. Tolkien",
            "William Faulkner",
            "Agatha Christie",
            "Maya Angelou",
            "Albert Camus",
            "Franz Kafka",
            "Simone de Beauvoir",
            "Margaret Atwood",
            "Stephen King",
            "Toni Morrison",
            "Harper Lee",
            "Kurt Vonnegut",
            "Oscar Wilde",
            "F. Scott Fitzgerald",
            "Ernest Rutherford",
            "Max Planck",
            "Niels Bohr",
            "Richard Feynman",
            "Stephen Hawking",
            "Alan Turing",
            "John von Neumann",
            "Carl Sagan",
            "Jane Goodall",
            "Rosalind Franklin",
            "Dorothy Hodgkin",
            "Elizabeth Blackwell",
            "Florence Nightingale",
            "Clara Barton",
            "Henry Ford",
            "Steve Jobs",
            "Bill Gates",
            "Elon Musk",
            "Jeff Bezos",
            "Warren Buffett",
            "Mark Zuckerberg",
            "Oprah Winfrey",
            "Walt Disney",
            "Coco Chanel",
            "Giorgio Armani",
            "Yves Saint Laurent",
            "Meryl Streep",
            "Audrey Hepburn",
            "Marilyn Monroe",
            "Elizabeth Taylor",
            "Katharine Hepburn",
            "Humphrey Bogart",
            "Marlon Brando",
            "Charlie Chaplin",
            "Buster Keaton",
            "Clint Eastwood",
            "Robert De Niro",
            "Al Pacino",
            "Tom Hanks",
            "Denzel Washington",
            "Leonardo DiCaprio",
            "Meryl Streep",
            "Julia Roberts",
            "Angelina Jolie",
            "Brad Pitt",
            "Johnny Depp",
            "George Clooney",
            "Will Smith",
            "Beyonce",
            "Michael Jackson",
            "Elvis Presley",
            "Madonna",
            "Whitney Houston",
            "Prince",
            "Bob Dylan",
            "The Beatles",
            "The Rolling Stones",
            "David Bowie",
            "Freddie Mercury",
            "Bruce Springsteen",
            "Taylor Swift",
            "Adele",
            "Justin Bieber",
            "Rihanna",
            "Drake",
            "Eminem",
            "Usain Bolt",
            "Michael Phelps",
            "Serena Williams",
            "Roger Federer",
            "Rafael Nadal",
            "Novak Djokovic",
            "Muhammad Ali",
            "Mike Tyson",
            "LeBron James",
            "Michael Jordan",
            "Kobe Bryant",
            "Lionel Messi",
            "Cristiano Ronaldo",
            "Diego Maradona",
            "Pelé",
            "Zinedine Zidane",
            "Tiger Woods",
            "Babe Ruth",
            "Jackie Robinson",
            "Joe DiMaggio",
            "Lou Gehrig",
            "Hank Aaron"
          ]
          indexes = random.sample(range(150), 3)
          p=[famous_names[indexes[0]],famous_names[indexes[1]],famous_names[indexes[2]]]
          return p
    else:
      most_similar = s2v.most_similar(sense, n=20)
      for each_word in most_similar:
          append_word = each_word[0].split("|")[0].replace("_", " ").lower()
          if append_word.lower() != word:
              output.append(append_word.title())
            
      out = list(OrderedDict.fromkeys(output))
      return out


question_tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')
question_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')

summary_model = T5ForConditionalGeneration.from_pretrained('t5-base')
summary_tokenizer = T5Tokenizer.from_pretrained('t5-base')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process(text):
  summarized_text = summarizer(text,summary_model,summary_tokenizer)
  imp_keywords = get_keywords(text,summarized_text)
  # print (imp_keywords)
  ques_ans = {}
  for answer in imp_keywords:
    choices = [" " for _ in range(5)]
    choices[random.randint(0,3)] = answer
    choices[4] = "Answer : "+ str(choices.index(answer)+1)
    ques = get_question(summarized_text,answer,question_model,question_tokenizer)
    distractors = sense2vec_get_words(answer, s2v)
    j = 0
    for i in range(4):
       if choices[i] == " ":
          choices[i] = distractors[j]
          j+=1
    ques_ans[ques] = choices
  # print(ques_ans)
  return ques_ans


def postprocesstext (content):
  final=""
  for sent in sent_tokenize(content):
    sent = sent.capitalize()
    final = final +" "+sent
  return final

def summarizer(text,model,tokenizer):
  text = text.strip().replace("\n"," ")
  text = "summarize: "+text
  max_len = 512
  encoding = tokenizer.encode_plus(text,max_length=max_len, pad_to_max_length=False,truncation=True, return_tensors="pt").to(device)
  input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
  outs = model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=3,
                                  num_return_sequences=1,
                                  no_repeat_ngram_size=2,
                                  # min_length =100,
                                  max_length=350)
  dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]
  summary = dec[0]
  summary = postprocesstext(summary)
  summary= summary.strip()
  return summary

def get_nouns_multipartite(content):
    out=[]
    try:
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(input=content,language='en')
        pos = {'PROPN','NOUN'}
        stoplist = list(string.punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        stoplist += stopwords.words('english')
        extractor.candidate_selection(pos=pos)
        extractor.candidate_weighting(alpha=1.1,
                                      threshold=0.75,
                                      method='average')
        keyphrases = extractor.get_n_best(n=15)
        for val in keyphrases:
            out.append(val[0])
    except:
        out = []
        traceback.print_exc()
    return out


def get_keywords(originaltext,summarytext):
  keywords = get_nouns_multipartite(originaltext)
  # print ("keywords unsummarized: ",keywords)
  keyword_processor = KeywordProcessor()
  for keyword in keywords:
    keyword_processor.add_keyword(keyword)

  keywords_found = keyword_processor.extract_keywords(summarytext)
  keywords_found = list(set(keywords_found))
  # print ("keywords_found in summarized: ",keywords_found)

  important_keywords =[]
  for keyword in keywords:
    if keyword in keywords_found:
      important_keywords.append(keyword)
  # indexes = random.sample(range(len(important_keywords)),4)
  # final_keywords = []
  # for index in indexes:
  #    final_keywords.append(important_keywords[index])
  # return final_keywords
  return important_keywords

def get_question(context,answer,model,tokenizer):
  text = "context: {} answer: {}".format(context,answer)
  encoding = tokenizer.encode_plus(text,max_length=384, pad_to_max_length=False,truncation=True, return_tensors="pt").to(device)
  input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
  outs = model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=5,
                                  num_return_sequences=1,
                                  no_repeat_ngram_size=2,
                                  max_length=72)
  dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]
  Question = dec[0].replace("question:","")
  Question= Question.strip()
  return Question

# process(input("Enter : "))
