#  ANS - Sztuczna Inteligencja (Laboratoria)

To miejsce służy jako centralny zbiór moich rozwiązań i projektów realizowanych w ramach zajęć laboratoryjnych z przedmiotu **Sztuczna Inteligencja**.

Wszystkie znajdujące się tutaj pliki (głównie skrypty w języku Python oraz notatniki `.ipynb` Jupyter/Google Colab) stanowią odpowiedź na zadania stawiane podczas poszczególnych etapów laboratoriów.

# Raport z Laboratorium nr 6: Rekurencyjne Sieci Neuronowe

## Cel projektu
Celem ćwiczenia było zaprojektowanie, implementacja oraz optymalizacja rekurencyjnych sieci neuronowych opartych na bramkach LSTM oraz GRU do prognozowania szeregów czasowych na przykładzie cen akcji firmy IBM. W ramach laboratorium przeprowadzono serię eksperymentów badających wpływ hiperparametrów na końcową precyzję modelu, ocenianą za pomocą metryki RMSE.

---

## Baza: Porównanie architektur LSTM i GRU 

W pierwszej fazie porównano bazowe konfiguracje sieci LSTM oraz GRU. Obie sieci składały się z 4 warstw ukrytych po 50 jednostek i były trenowane z batch size równym 32.

### Prędkość uczenia
* **LSTM:** Architektura LSTM wymagała dłuższego czasu na wykonanie pojedynczej epoki treningowej. Wynika to ze złożonej struktury wewnętrznej komórki, która posiada trzy bramki: wejściową, wyjściową i zapominania.
* **GRU:** Architektura GRU charakteryzowała się wyższą efektywnością obliczeniową. Krótszy czas epoki jest bezpośrednim rezultatem uproszczonej konstrukcji komórki, która integruje operacje w dwie bramki: aktualizacji i resetowania.

### Logi treningu i Wykresy Predykcji
Poniższe zrzuty ekranu przedstawiają proces uczenia obu sieci oraz ich końcowe dopasowanie do danych testowych.

#### 1. Model LSTM
<img width="441" height="349" alt="LSTM (1)" src="https://github.com/user-attachments/assets/30c50f5c-516f-45dc-9023-80cc651c5a41" />
*Rys 1. Logi z przebiegu uczenia bazowej sieci LSTM.*

<img width="335" height="104" alt="LSTM (2)" src="https://github.com/user-attachments/assets/5ac82bbb-1873-4d8a-94e5-5f98847add91" />
*Rys 2. Porównanie rzeczywistych cen akcji IBM z predykcją modelu LSTM.*

#### 2. Model GRU
<img width="438" height="380" alt="GRU (3)" src="https://github.com/user-attachments/assets/bcb115e3-5b91-462f-9455-79a60be21927" />
*Rys 3. Logi z przebiegu uczenia bazowej sieci GRU.*

<img width="326" height="109" alt="GRU (1)" src="https://github.com/user-attachments/assets/dc785852-23e2-4527-b662-ebc9662f1b23" />
*Rys 4. Porównanie rzeczywistych cen akcji IBM z predykcją modelu GRU.*

---

## A. Eksperymenty z ilością jednostek LSTM

Przeprowadzono modyfikacje liczby neuronów w warstwach ukrytych sieci LSTM, testując skrajne warianty:
* **Niska pojemność (units=20):** Zaobserwowano zjawisko niedouczenia (underfitting). Predykcja była zbytnio wygładzona, przez co sieć nie była w stanie poprawnie odwzorować dynamicznych zmian trendu.
* **Wysoka pojemność (units=128):** Sieć bardzo dobrze dopasowała się do danych treningowych, jednak znacząco wydłużyło to czas obliczeń. Zbyt duża liczba parametrów bez odpowiedniej regularyzacji zwiększyła ryzyko przeuczenia (overfittingu) na szumie giełdowym. 
* **Wniosek:** Optymalnym kompromisem dla tego zestawu danych okazała się wartość bazowa w przedziale 50-64 jednostek na warstwę.

---

## B i C. Analiza optymalizatorów i funkcji straty

W kolejnym etapie zbadano wpływ algorytmów optymalizacji oraz metod obliczania błędu na proces aktualizacji wag sieci.

### 1. Test optymalizatora SGD
Zastosowanie klasycznego stochastycznego spadku wzdłuż gradientu (SGD) bez zaawansowanego dostrajania parametrów  okazało się mało efektywne. Model utknął w minimum lokalnym, co skutkowało wysokim błędem średniokwadratowym oraz prostoliniowym, nienaturalnym wykresem predykcji.

<img width="639" height="484" alt="SGD (2)" src="https://github.com/user-attachments/assets/5b97e090-05df-4eb2-9bea-b1b2249ea38e" />
*Rys 5. Rezultat dopasowania modelu przy użyciu optymalizatora SGD.*

### 2. Połączenie optymalizatora Adam i funkcji straty Huber
Najwyższą dokładność uzyskano po zastąpieniu domyślnego optymalizatora `rmsprop` i funkcji `mean_squared_error` algorytmem **Adam** oraz funkcją straty **Huber**.
* **Adam:** Jako optymalizator adaptacyjny, poprawnie dobiera oddzielne kroki uczenia dla poszczególnych wag, co znacząco przyspiesza zbieżność modelu.
* **Huber Loss:** Funkcja ta łączy cechy MSE i MAE. Dla małych błędów zachowuje się jak błąd kwadratowy, a dla dużych - liniowo. Sprawia to, że model jest bardziej odporny na tzw. szum oraz nagłe piki cenowe typowe dla rynków finansowych, zapobiegając rozregulowaniu wag w sieci.

<img width="640" height="484" alt="adam_huber (2)" src="https://github.com/user-attachments/assets/0d7a652d-fd4b-4b5d-8192-37416428d89d" />
*Rys 6. Dopasowanie modelu wykorzystującego konfigurację Adam + Huber Loss.*

---

## D. Zmiana atrybutu predykcji z „High” na „Close”

Zmieniono kolumnę docelową z dziennej ceny maksymalnej (`High` - indeks 1) na cenę zamknięcia sesji (`Close` - indeks 3), modyfikując wywołania wycinania danych na `.iloc[:, 3:4]`.

**Obserwacje:** Prognozowanie cen zamknięcia ma nieco inną dynamikę. Ceny te są wypadkową działań inwestorów na koniec dnia sesyjnego, co generuje inną strukturę trendu niż w przypadku dziennych maksimów. Sieć poprawnie zaadaptowała się do nowego celu, utrzymując stabilność, jednak potwierdziło to odmienną specyfikację tych dwóch atrybutów.

<img width="643" height="479" alt="3_4_Close (2)" src="https://github.com/user-attachments/assets/8981a86e-bea2-4b49-a1d6-646f2ac73d6f" />
*Rys 7. Wynik predykcji dla prognozowania cen zamknięcia (Close).*

---

## E. Badanie efektywności mechanizmu Early Stopping i walidacji

W celu zabezpieczenia modelu przed zjawiskiem przeuczenia wdrożono mechanizm wczesnego zatrzymywania za pomocą klasy callback `EarlyStopping`. Równocześnie wprowadzono podział danych wejściowych, wydzielając 10% zbioru treningowego jako niezależną próbę walidacyjną (`validation_split=0.1`). W przeciwieństwie do standardowego podejścia monitorującego funkcję straty na danych uczących, proces optymalizacji i ewaluacji oparto na obserwacji wartości błędu walidacyjnego (`monitor='val_loss'`).

Maksymalną liczbę epok zdefiniowano pierwotnie na 100. Podczas eksperymentu zaobserwowano, że dzięki zastosowaniu mechanizmu Early Stopping, algorytm optymalizacyjny samodzielnie przerwał procedurę uczenia już na 10. epoce. Nastąpiło to w momencie, gdy błąd na zbiorze walidacyjnym przestał ulegać poprawie. Dalsza nauka skutkowałaby jedynie zapamiętywaniem próbek uczących i spadkiem zdolności modelu do generalizacji trendów giełdowych. Wdrożenie tej metody uchroniło sieć przed przeuczeniem oraz znacząco zoptymalizowało czas trwania obliczeń.

<img width="623" height="140" alt="val_loss_earlystop" src="https://github.com/user-attachments/assets/041f2d46-201d-4a37-ba14-62bf18750bbf" />
*Rys. 8. Przebieg procesu uczenia przerwany automatycznie na 10. epoce przez mechanizm Early Stopping na podstawie metryki val_loss.*

---

## F i G. Podsumowanie i Ostateczna Konfiguracja (RMSE < 2.0)

Eksperymenty pokazały, że samo zwiększanie złożoności strukturalnej sieci nie gwarantuje lepszych wyników bez odpowiedniej optymalizacji hiperparametrów i zastosowania technik regularyzacji.

### Ostateczna konfiguracja
* **Architektura:** LSTM / GRU (dobierane zależnie od preferencji czas obliczeń vs nieznaczna poprawa stabilności)
* **Liczba jednostek:** 50 - 64 na warstwę
* **Optymalizator:** `Adam`
* **Funkcja straty (Loss):** `Huber`
* **Regularyzacja:** `EarlyStopping(monitor='val_loss', patience=5)` oraz `Dropout(0.2)`

### Uzasadnienie
Zastosowanie optymalizatora adaptacyjnego (Adam) w połączeniu z funkcją błędu uodpornioną na anomalie rynkowe (Huber) doprowadziło do znacznej redukcji błędu średniokwadratowego. Wdrożenie podziału walidacyjnego wraz z mechanizmem Early Stopping zapewniło, że nauczony model posiada faktyczną zdolność do generalizacji trendów, zamiast jedynie odtwarzać dane z próby treningowej.
