import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

df_full = pd.read_csv('dataset/features/extracted_features_17_classes.csv')

bird_species = os.listdir('dataset/17_classes_audio')
print(df_full)

num_species = len(bird_species)     # liczba gatunków
num_samples = 200    # liczba probek na kazda klase
num_features = len(df_full.columns) - 1      # liczba wyekstraktowanych cech dla jednego pliku

X = df_full.iloc[:, 1:].values
y = df_full.iloc[:, 0].values

# standaryzacja
X = StandardScaler().fit_transform(X, y)

# redukcja wymiaru do 2 cech
X = LDA(n_components=(num_species-1)).fit_transform(X, y)

df_full.to_csv('dataset/features/reduced_features_17_classes.csv', index=False)

classes = np.linspace(0, len(df_full.index), num_species + 1)

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20,14))

# wykres zaleznosci cech po redukcji wymiaru
classes = np.linspace(0, len(df_full.index), num_species + 1)
figure, axs =plt.subplots()
axs = figure.add_subplot(projection='3d')
for i in classes:
    if i < len(df_full.index):
        i = int(i)
        ax[0][0].scatter(X[int(i):int(i+num_samples), 0], X[int(i):int(i+num_samples), 1])
        axs.scatter(X[int(i):int(i+num_samples), 0], X[int(i):int(i+num_samples), 1], X[int(i):int(i+num_samples), 2])
ax[0][0].set_title('Wykres zaleznosci dwóch pierwszych zredukowanych cech')
ax[0][0].legend(bird_species, loc='upper right')
axs.set_title('Wykres 3d zależności trzech pierwszych zredukowanych cech.')


# dzielenie danych
X_train, X_test, y_train, y_test = train_test_split(X, y)


# SVM

clf_svm = SVC()
clf_svm.fit(X_train, y_train)
pred_svm = clf_svm.predict(X_test)
acc_svm = accuracy_score(y_test, pred_svm)
report_svm = classification_report(y_test, pred_svm, target_names=bird_species)
# df_svm = pd.DataFrame(report_svm).transpose()
# df_svm.to_csv('results/classification_reports/classification_report_svm_17.csv')

print('Dokładność SVM: ', acc_svm)
print('SVM:', report_svm)
print(f1_score(y_test, pred_svm, average='micro'))


ax[0][1].set_title('Macierz pomyłek dla SVD')
confusion_matrix_svm = confusion_matrix(y_test, pred_svm)
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_svm, display_labels=clf_svm.classes_).plot(cmap='cividis', xticks_rotation='vertical', ax=ax[0, 1])
# ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_svm, display_labels=clf_svm.classes_).plot(cmap='cividis', xticks_rotation='vertical', ax=axes)

# Drzewo decyzyjne

clf_tree = DecisionTreeClassifier()
clf_tree.fit(X_train, y_train)
pred_tree = clf_tree.predict(X_test)
acc_tree = accuracy_score(y_test, pred_tree)
report_dc = classification_report(y_test, pred_tree, target_names=bird_species)
# df_dc = pd.DataFrame(report_dc).transpose()
# df_dc.to_csv('results/classification_reports/classification_report_dc_17.csv')


print('Dokładność drzewa decyzyjnego: ', acc_tree)
print('DC:', report_dc)
print(f1_score(y_test, pred_tree, average='micro'))



ax[1][0].set_title('Macierz pomyłek dla DC')
confusion_matrix_tree = confusion_matrix(y_test, pred_tree)
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_tree, display_labels=clf_svm.classes_).plot(cmap='cividis', xticks_rotation='vertical', ax=ax[1][0])
# ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_tree, display_labels=clf_svm.classes_).plot(cmap='cividis', xticks_rotation='vertical')


# K-najblizszych sasiadow

clf_knn = KNeighborsClassifier()
clf_knn.fit(X_train, y_train)
pred_knn = clf_knn.predict(X_test)
acc_knn = accuracy_score(y_test, pred_knn)
report_knn = classification_report(y_test, pred_knn, target_names=bird_species)
# df_knn = pd.DataFrame(report_knn).transpose()
# df_knn.to_csv('results/classification_reports/classification_report_knn_17.csv')

print('Dokładność k-najblizszych sasiadow: ', acc_knn)
print('KNN:', report_knn)
print(f1_score(y_test, pred_knn, average='micro'))


ax[1, 1].set_title('Macierz pomyłek dla KNN')
confusion_matrix_knn = confusion_matrix(y_test, pred_knn)
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_knn, display_labels=clf_svm.classes_).plot(cmap='cividis', xticks_rotation='vertical', ax=ax[1, 1])
# ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_knn, display_labels=clf_svm.classes_).plot(cmap='cividis', xticks_rotation='vertical')


plt.show()
