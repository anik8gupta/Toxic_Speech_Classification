# Import Libraries
from flask import Flask,render_template,url_for,request
import pandas as pd 

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.externals import joblib


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	 
    import pickle as p
    # un-serializing model
    clf1 = p.load(open('speech_classification1.pkl', 'rb'))
    clf2 = p.load(open('speech_classification2.pkl', 'rb'))
    clf3 = p.load(open('speech_classification3.pkl', 'rb'))
    clf4 = p.load(open('speech_classification4.pkl', 'rb'))
    clf5 = p.load(open('speech_classification5.pkl', 'rb'))
    clf6 = p.load(open('speech_classification6.pkl', 'rb'))
    
    message = request.form['message']
    data = message

    import re
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer

    ps = PorterStemmer()
    #getting setences from speech#
    from nltk.tokenize import sent_tokenize
    tokenize=sent_tokenize(data)

    corpus3=[]

    for i in range(0, len(tokenize)):
        review3 = re.sub('[^a-zA-Z]', ' ', tokenize[i])
        review3 = review3.lower()
        review3 = review3.split()
        #review = [word for word in review if not word in set(stopwords.words('english'))]
        review3 = [ps.stem(word) for word in review3 if not word in set(stopwords.words('english'))]
        review3 = ' '.join(review3)
        corpus3.append(review3)

    #getting best 100 words
    cv3 = CountVectorizer(max_features = 100)
    X3 = cv3.fit_transform(corpus3).toarray()
    
    #predicting
    y_pred1 = clf1.predict(X3)
    y_pred2 = clf2.predict(X3)
    y_pred3 = clf3.predict(X3)
    y_pred4 = clf4.predict(X3)
    y_pred5 = clf5.predict(X3)
    y_pred6 = clf6.predict(X3)

    #conveting them in Data Frame
    y_pred1_df=pd.DataFrame(y_pred1)
    y_pred2_df=pd.DataFrame(y_pred2)
    y_pred3_df=pd.DataFrame(y_pred3)
    y_pred4_df=pd.DataFrame(y_pred4)
    y_pred5_df=pd.DataFrame(y_pred5)
    y_pred6_df=pd.DataFrame(y_pred6)


    f=y_pred6_df.iloc[:,0].values
    f2=y_pred5_df.iloc[:,0].values
    f3=y_pred4_df.iloc[:,0].values
    f4=y_pred3_df.iloc[:,0].values
    f5=y_pred2_df.iloc[:,0].values
    f6=y_pred1_df.iloc[:,0].values

    #making a final Submission Data frame
    submission = pd.DataFrame({'id':corpus3,'toxic':f,'severe_toxic':f2,
                           'obscene':f3,
                           'threat':f4,
                           'insult':f5,
                           'identity_hate':f6})

    #getting total of all rows#
    submission['total']=submission.sum(axis=1)
        
    #creating a normal column#
    a=[]
    for row in submission['total']:
        if row==0:
            a.append(1)
        else:
            a.append(0)
    submission['normal']=pd.DataFrame(a)

    #getting total of column#
    total=submission[['toxic','severe_toxic','obscene','threat','insult','identity_hate','normal']].sum()
    
    
    import matplotlib.pyplot as plt
    import io
    import base64
    import urllib
    #making and saving pie-chart
    img = io.BytesIO() 
    plt.pie(total)
    plt.title("pie chart distribution")
    plt.savefig(img, format='png')
    img.seek(0)

    plot_data = urllib.parse.quote(base64.b64encode(img.read()).decode())

    #returning results with requested html page	
    return render_template('result.html',normal=(total[6]/total.sum())*100,
                           toxic=(total[0]/total.sum())*100,
                           severe_toxic=(total[1]/total.sum())*100,
                           obscene=(total[2]/total.sum())*100,
                           threat=(total[3]/total.sum())*100,
                           insult=(total[4]/total.sum())*100,
                           identity_hate=(total[5]/total.sum())*100,plot_url=plot_data)



if __name__ == '__main__':
	app.run(debug=True)
