import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud 
from sklearn.metrics import confusion_matrix, classification_report


def distrubution_show (df):
    img_dir_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
    if not os.path.exists(img_dir_save_path):
        os.makedirs(img_dir_save_path)

    plt.figure(figsize = (15, 15))
    sns.countplot(y = df['Cat1'], data = df)
    plt.tick_params(labelsize=10)
    plt.title('Category 1 Distribution', fontsize=20)
    plt.xlabel("The Number of Messages", fontsize=14)
    plt.ylabel("Categoty 1", fontsize=14)
    plt.savefig(os.path.join(os.path.dirname(__file__), '../images', 'cat1_distrubution.png'))

    plt.figure(figsize = (12, 12))
    sns.countplot(y = df['Cat2'], data = df)
    plt.tick_params(labelsize=8)
    plt.title('Category 2 Distribution', fontsize=20)
    plt.xlabel("The Number of Messages", fontsize=14)
    plt.ylabel("Categoty 2", fontsize=14)
    plt.savefig(os.path.join(os.path.dirname(__file__), '../images', 'cat2_distrubution.png'))


def word_cloud_show(vocabulary):
    vocabulary = ' '.join([str(x) for x in vocabulary.keys()])
    wordcloud = WordCloud(background_color="white", max_words =500).generate(vocabulary)

    img_dir_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
    if not os.path.exists(img_dir_save_path):
        os.makedirs(img_dir_save_path)

    plt.figure(figsize = (20, 20))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig(os.path.join(os.path.dirname(__file__), '../images', 'word_cloud.png')) 

def word_cloud_by_classes_show(X_train, y_train):

    img_dir_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../images', 'word_clouds_by_class')
    if not os.path.exists(img_dir_save_path):
        os.makedirs(img_dir_save_path)

    y_train = pd.Series(y_train, name='Classes')
    df = pd.concat([X_train, y_train], axis=1)

    labels = y_train.unique()
    for l in labels:
        df_class = ' '.join([str(x) for x in df[df['Classes'] == l]['Text']])
        wordcloud = WordCloud(background_color="white", max_words =500).generate(df_class)

        plt.figure(figsize = (20, 20))
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.savefig(os.path.join(os.path.dirname(__file__), '../images/word_clouds_by_class', 
            l.replace(' ', '_').replace('/', '&') + '.png')) 
        plt.close()


def quality_metrics_show(y_test, y_pred, level):
    img_dir_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
    if not os.path.exists(img_dir_save_path):
        os.makedirs(img_dir_save_path)

    cm=confusion_matrix(y_test, y_pred)
    #print(classification_report(y_test, y_pred, digits=3))
    cm_df=pd.DataFrame(cm)
    sns.set(font_scale=1.4,color_codes=True,palette="deep")
    if level == 0:
        sns.heatmap(cm_df,annot=True,annot_kws={"size":12},fmt="d",cmap="YlGnBu")
    else:
        plt.figure(figsize = (20, 20))
        if level == 1:
            sns.heatmap(cm_df,annot=True,annot_kws={"size":20},fmt="d",cmap="BuPu")
        else:
            sns.heatmap(cm_df,annot=True,annot_kws={"size":12},fmt="d",cmap="BuPu")
            #plt.axis("off")
        classes = sorted(y_test.unique())
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes, rotation=360)
    
    plt.title("Confusion Matrix", fontsize=20)
    plt.xlabel("Predicted Value", fontsize=16)
    plt.ylabel("True Value", fontsize=16)
    plt.savefig(os.path.join(os.path.dirname(__file__), '../images', 'confusion_matrix_' + str(level) + '.png')) 
    plt.close()
