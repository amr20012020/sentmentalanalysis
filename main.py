# pip install flask
# pip install joblib
# pip install sklearn
# pip install numpy
# pip install pandas


from flask import Flask , jsonify , request
import joblib

import numpy as np # hna ana b3ml import lnumpy ely bst5dmha fi ay math operation
import pandas as pd # hna ana b3ml import lpandas ely bst5dmha fi ay data analysis process


# dah code lazm ytnfz el awl 3alashan ast5dm el stopwords w el word_tokenize
import nltk
nltk.download("punkt");

# instantiate the Maximum Likelihood Disambiguator
#mle = MLEDisambiguator.pretrained()

import re # hna ana b3ml import l library el regular expression 3alashan asheel beha el punctuation ely fi el reviews
import string
from nltk import word_tokenize # hna ana b3ml import el natural language processing library ely esmha nltk w bstd3y mnha el word tokenizer



class SentimentAnalysis:


    def remove_punct(self , review):
      punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ''' + string.punctuation;
      translator = str.maketrans('', '', punctuations);
      cleaned_review = review.translate(translator);
      return cleaned_review;

    def remove_arabic_diacritics(self , text):
      arabic_diacritics = re.compile("""
                            ّ    | # Shadda
                            َ    | # Fatha
                            ً    | # Tanwin Fath
                            ُ    | # Damma
                            ٌ    | # Tanwin Damm
                            ِ    | # Kasra
                            ٍ    | # Tanwin Kasr
                            ْ    | # Sukun
                            ـ     # Tatwil/Kashida
                        """, re.VERBOSE); # re.verbose dah flag bysm7ly eny aktb regular expressions bshkl readable aktr w bysm7ly eny aktb comment 3ala kol value
      cleaned_text = re.sub(arabic_diacritics , "" , text);
      return cleaned_text;

    def text_normalization(self , text):
      text = re.sub(r"[a-zA-Z]", "", text);
      text = re.sub(r"\n+", " ", text);
      text = re.sub(r"\t+", "", text);
      text = re.sub(r"\r+", " ", text);
      text = re.sub(r"\s+", " ", text);
      text = re.sub("[إأٱآا]", "ا", text);
      text = re.sub("ى", "ي", text);
      text = re.sub("ؤ", "ء", text);
      text = re.sub("ئ", "ء", text);
      text = re.sub("ة", "ه", text);
      text = re.sub("گ", "ك", text);
      text = re.sub(r'\d+', "", text);
      return text;

    def remove_duplicates(self , tokens_list):
      new_tokens_list = [];
      for token in tokens_list: # ha3mlha comment b3den
        # dict.fromkeys() di function bt3ml dictionary w bt3tbr el keys aw el index bta3tha hya el list of tokens di (hna ana basheel el chars el mtkrra)
        new_tokens_list.append(self.remove_repeating_char(token));

      new_tokens_list_size = len(new_tokens_list);
      for i in range(new_tokens_list_size-1):
        if(new_tokens_list[i] == new_tokens_list[i+1]):
          new_tokens_list[i] = "";
        
      return word_tokenize(" ".join(new_tokens_list)); # (hna ana basheel el kalamat el mtkrra)    

    def stemming_review(self , tokens_list):
      new_token_list = [];
      ar_stemmer = stemmer("arabic");
      for token in tokens_list:
        new_token_list.append(ar_stemmer.stemWord(token));
      return new_token_list;

    def create_stopwords_list(self):
      stopwords_list = [];
      with open('stopwords/stopwords.txt', "r" , encoding='utf-8') as stopwords_file: # with 3alashan a3ml close llfile automatically mn 8air ma anady 3ala el close() func
          for line in stopwords_file:
            stopwords_list.append(line.rstrip("\n"));
      
      with open('stopwords/final_stop_words.txt' , "r" , encoding='utf-8') as stopwords_file:
        for line in stopwords_file:
          stopwords_list.append(line.rstrip("\n"));

      temp_list = [];

      for word in stopwords_list:
        new_word = self.remove_arabic_diacritics(word);
        new_word = self.text_normalization(new_word);
        temp_list.append(new_word);
      stopwords_list = list(dict.fromkeys(temp_list));
      return stopwords_list;

    def remove_stopwords(self , tokens_list):
      stopwords_list = self.create_stopwords_list()
      new_tokens_list = [];
      for token in tokens_list:
        if(not token in stopwords_list):
          new_tokens_list.append(token);
      return new_tokens_list;

    def remove_repeating_char(self , text):
        return re.sub(r'(.)\1+', r'\1', text);

    def text_preprocessing(self , review):
      # wo stands for without
      review_wo_punct = self.remove_punct(review);

      review_wo_diacritics = self.remove_arabic_diacritics(review_wo_punct);

      normalized_review = self.text_normalization(review_wo_diacritics);
      review_tokens = word_tokenize(normalized_review);
      review_wo_duplicates = self.remove_duplicates(review_tokens);

      review_wo_stopwords = self.remove_stopwords(review_wo_duplicates);

      #stemmed_review = stemming_review(review_wo_stopwords); # review_wo_stopwords

      return " ".join(review_wo_stopwords); # return list of tokens

app = Flask(__name__);
clf_model = joblib.load("nlpmodel/mymodel.pkl");

@app.route("/" , methods=["POST"])
def get_my_doc():
    if(request.method == "POST"):
        d = {};
        json_obj = request.json;
        incoming_txt = json_obj["textanalysis"];
        print("incoming txt : ", incoming_txt);
        sent = SentimentAnalysis();
        cleaned_txt = sent.text_preprocessing(incoming_txt);
        print("cleaned txt : " ,cleaned_txt); 
        result = str(clf_model.predict([cleaned_txt])[0]);
        print("result : " , result);
        d["result"] = result;
        return jsonify(d);
        
if(__name__ == "__main__"):
    app.run();
