
# DBE-Angleichungsmodul-Aufgabe2

# Ziele der Aufgabe:
- Implementierung von ML basierten SW Systemen
- Betrieb und Ablauf von ML basierter Software transparent gestalten
- Automatisches Testen von ML basierten SW Systemen

# Ausführen
1. Öffnen Sie den Link zu Binder, den Sie im Binder Badge finden.
Um eine .py-Datei in Binder auszuführen, starten Sie ein neues Terminal innerhalb der Binder-Umgebung.
2. Führen Sie zunächst die main.py-Datei aus, indem Sie den Befehl "python src/main.py" eingeben. Das Ergebnis wird in der Testdatendatei im Unterverzeichnis data gespeichert. Wenn Sie die Ausgabe in eine .txt-Datei exportieren möchten, verwenden Sie den Befehl "python src/main.py >> ausgabe.txt". Sie finden die ausgabe.txt-Datei im Hauptverzeichnis.
3. Führen Sie die test.py-Datei aus, indem Sie den Befehl "python tests/test.py" verwenden. Um die Ausgabe in eine .txt-Datei zu exportieren, nutzen Sie den Befehl "python tests/test.py >> ausgabe2.txt". Die ausgabe2.txt-Datei befindet sich ebenfalls im Hauptverzeichnis.
4. Führen Sie die main_runtime.py aus, indem Sie Befehl "python src/main_runtime.py" eingeben. Das Ergebnis wird in der baseline_runtime.txt im Unterverzeichnis data gespeichert. 
5. Führen Sie die Datei test_runtime.py aus, indem Sie den Befehl "python tests/test_runtime.py" verwenden. Der Schwellenwert für den Laufzeittest ergibt sich aus den Durchschnittswerten, die aus der Datei baseline_runtime.txt gelesen werden. Im Falle eines Fehlers wird eine Fehlermeldung ausgegeben, die besagt, dass die Laufzeit den Schwellenwert überschritten hat.
Um die Ausgabe in eine .txt-Datei zu exportieren, nutzen Sie den Befehl "python tests/test_runtime.py >> ausgabe3.txt". Die ausgabe3.txt-Datei befindet sich ebenfalls im Hauptverzeichnis.

# Erwartete Ergebnis:
1. Ergebnis nach der Ausführung der main.py Datei ist in der Datei ausgabe.txt zu finden.
   
[ausgabe.txt](https://github.com/hienchipham197/DBE-Angleichungsmodul-Aufgabe2/files/14343872/ausgabe.txt)

MNIST: (70000, 784) (70000,)
__init__ ran in: 6.67572021484375e-06 sec
fit ran in: 25.985992193222046 sec

Train Accuracy: 72.65166666666667 

Train confusion matrix:
[[5465    7   31   25   45   18  196   59   71    6]
 [   3 6481   98   45    2   29   25   40   18    1]
 [ 339  478 3603  165  286   19  689  210  115   54]
 [ 143  255  224 4521   60  256  102  146  241  183]
 [ 106  136   23   51 4521  331  198  150   84  242]
 [ 414  214   85 1032  402 2261  292  408  186  127]
 [ 189   96   91   53  157  230 5038   29   35    0]
 [ 209  203  183   38  142   16   25 5108   50  291]
 [  72  770  181  692   72  191  352   48 3317  156]
 [ 167  182   48  257  625  265  120  782  227 3276]]

Classification report for classifier:
               precision    recall  f1-score   support

           0       0.78      0.94      0.85       980
           1       0.76      0.96      0.85      1135
           2       0.80      0.62      0.69      1032
           3       0.67      0.76      0.71      1010
           4       0.71      0.76      0.74       982
           5       0.62      0.41      0.50       892
           6       0.69      0.84      0.76       958
           7       0.74      0.81      0.77      1028
           8       0.77      0.60      0.67       974
           9       0.77      0.57      0.65      1009

    accuracy                           0.73     10000
   macro avg       0.73      0.73      0.72     10000
weighted avg       0.73      0.73      0.72     10000


predict ran in: 1.9965155124664307 sec

Test Accuracy: 73.18 

Test confusion matrix:
[[ 922    2    2    3    3    1   36    4    7    0]
 [   0 1090   18   11    0    0    5    5    6    0]
 [  66   94  636   30   40    2   98   33   25    8]
 [  21   30   31  766    7   44   18   32   38   23]
 [  15   20    2    8  748   58   45   26   18   42]
 [  61   27   12  185   74  368   58   65   23   19]
 [  36   10   16    8   26   42  804    8    7    1]
 [  23   43   47    6   16    3    6  830   10   44]
 [  14   98   27  100   10   33   70    9  583   30]
 [  21   29    9   33  129   43   28  108   38  571]]


2. Ergebnis nach der Ausführung der test.py Datei ist in der Datei ausgabe2.txt zu finden.
   
[ausgabe2.txt](https://github.com/hienchipham197/DBE-Angleichungsmodul-Aufgabe2/files/14343874/ausgabe2.txt)

setUp
__init__ ran in: 2.765655517578125e-05 sec
fit ran in: 26.321918725967407 sec
setUp
__init__ ran in: 5.7220458984375e-06 sec
fit ran in: 25.157458305358887 sec
Classification report for classifier:
               precision    recall  f1-score   support

           0       0.78      0.94      0.85       980
           1       0.76      0.96      0.85      1135
           2       0.80      0.62      0.69      1032
           3       0.67      0.76      0.71      1010
           4       0.71      0.76      0.74       982
           5       0.62      0.41      0.50       892
           6       0.69      0.84      0.76       958
           7       0.74      0.81      0.77      1028
           8       0.77      0.60      0.67       974
           9       0.77      0.57      0.65      1009

    accuracy                           0.73     10000
   macro avg       0.73      0.73      0.72     10000
weighted avg       0.73      0.73      0.72     10000


predict ran in: 2.099426746368408 sec

3. Ergebnis nach der Ausführung der main_runtime.py Datei:

init__ ran in: 1.2874603271484375e-05 sec
fit ran in: 26.45095992088318 sec
Die Laufzeit des fit()-Aufrufs wurde an data/baseline_runtime.txt angehängt: 26.45107412338257 Sekunden

4. Ergebnis nach der Ausführung der test_runtime.py Datei --> keine Fehlermeldung

Setting up class for runtime tests.

Setup for runtime test

__init__ ran in: 5.9604644775390625e-06 sec

fit ran in: 26.31524682044983 sec

.
----------------------------------------------------------------------
Ran 1 test in 37.876s

# Binder Badge
Das Prjekt kann in Binder ausgeführt werden: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/hienchipham197/DBE-Angleichungsmodul-Aufgabe2/HEAD)
