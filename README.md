#  ANS - Sztuczna Inteligencja (Laboratoria)

To miejsce służy jako centralny zbiór moich rozwiązań i projektów realizowanych w ramach zajęć laboratoryjnych z przedmiotu **Sztuczna Inteligencja**.

Wszystkie znajdujące się tutaj pliki (głównie skrypty w języku Python oraz notatniki `.ipynb` Jupyter/Google Colab) stanowią odpowiedź na zadania stawiane podczas poszczególnych etapów laboratoriów.

# Raport z Laboratorium nr 6: Rekurencyjne Sieci Neuronowe (LSTM vs GRU)

## Cel projektu
Celem ćwiczenia było zaprojektowanie, implementacja oraz optymalizacja rekurencyjnych sieci neuronowych (RNN) opartych na bramkach LSTM oraz GRU do prognozowania szeregów czasowych na przykładzie cen akcji firmy IBM. W ramach laboratorium przeprowadzono serię eksperymentów badających wpływ hiperparametrów (liczby jednostek, optymalizatorów, funkcji straty, doboru cech oraz mechanizmu wczesnego zatrzymywania) na końcową precyzję modelu, ocenianą za pomocą metryki RMSE.

---

## Baza: Porównanie architektur LSTM i GRU (prędkość uczenia)

W pierwszej fazie porównano bazowe konfiguracje sieci LSTM oraz GRU. Obie sieci składały się z 4 warstw ukrytych po 50 jednostek i były trenowane z rozmiarem paczki (batch size) równym 32.

### Prędkość uczenia
* **LSTM:** Architektura LSTM wymagała dłuższego czasu na wykonanie pojedynczej epoki treningowej. Wynika to ze złożonej struktury wewnętrznej komórki, która posiada trzy bramki (wejściową, wyjściową i zapominania).
* **GRU:** Architektura GRU charakteryzowała się wyższą efektywnością obliczeniową. Krótszy czas epoki jest bezpośrednim rezultatem uproszczonej konstrukcji komórki, która integruje operacje w dwie bramki (aktualizacji i resetowania).

### Logi treningu i Wykresy Predykcji
Poniższe zrzuty ekranu przedstawiają proces uczenia obu sieci oraz ich końcowe dopasowanie do danych testowych.

#### 1. Model LSTM
![Logi treningu LSTM](LSTM%20(1).png)
*Rys 1. Logi z przebiegu uczenia bazowej sieci LSTM.*

![Wykres predykcji LSTM](LSTM%20(2).png)
*Rys 2. Porównanie rzeczywistych cen akcji IBM z predykcją modelu LSTM.*

#### 2. Model GRU
![Logi treningu GRU](GRU%20(1).png)
*Rys 3. Logi z przebiegu uczenia bazowej sieci GRU.*

![Wykres predykcji GRU](GRU%20(3).png)
*Rys 4. Porównanie rzeczywistych cen akcji IBM z predykcją modelu GRU.*

---

## A. Eksperymenty z ilością jednostek LSTM (Units)

Przeprowadzono modyfikacje liczby neuronów w warstwach ukrytych sieci LSTM, testując skrajne warianty:
* **Niska pojemność (units=20):** Zaobserwowano zjawisko niedouczenia (underfitting). Predykcja była zbytnio wygładzona, przez co sieć nie była w stanie poprawnie odwzorować dynamicznych zmian trendu.
* **Wysoka pojemność (units=128):** Sieć bardzo dobrze dopasowała się do danych treningowych, jednak znacząco wydłużyło to czas obliczeń. Zbyt duża liczba parametrów bez odpowiedniej regularyzacji zwiększyła ryzyko przeuczenia (overfittingu) na szumie giełdowym. 
* **Wniosek:** Optymalnym kompromisem dla tego zestawu danych okazała się wartość bazowa w przedziale 50-64 jednostek na warstwę.

---

## B i C. Analiza optymalizatorów i funkcji straty

W kolejnym etapie zbadano wpływ algorytmów optymalizacji oraz metod obliczania błędu na proces aktualizacji wag sieci.

### 1. Test optymalizatora SGD
Zastosowanie klasycznego stochastycznego spadku wzdłuż gradientu (SGD) bez zaawansowanego dostrajania parametrów (takich jak współczynnik uczenia czy momentum) okazało się mało efektywne. Model utknął w minimum lokalnym, co skutkowało wysokim błędem średniokwadratowym oraz prostoliniowym, nienaturalnym wykresem predykcji.

![Wykres predykcji SGD](SGD%20(1).png)
*Rys 5. Rezultat dopasowania modelu przy użyciu optymalizatora SGD.*

### 2. Połączenie optymalizatora Adam i funkcji straty Huber
Najwyższą dokładność uzyskano po zastąpieniu domyślnego optymalizatora `rmsprop` i funkcji `mean_squared_error` algorytmem **Adam** oraz funkcją straty **Huber**.
* **Adam:** Jako optymalizator adaptacyjny, poprawnie dobiera oddzielne kroki uczenia dla poszczególnych wag, co znacząco przyspiesza zbieżność modelu.
* **Huber Loss:** Funkcja ta łączy cechy MSE i MAE. Dla małych błędów zachowuje się jak błąd kwadratowy, a dla dużych - liniowo. Sprawia to, że model jest bardziej odporny na tzw. szum oraz nagłe piki cenowe typowe dla rynków finansowych, zapobiegając rozregulowaniu wag w sieci.

![Predykcja Adam + Huber](adam_huber%20(1).png)
*Rys 6. Dopasowanie modelu wykorzystującego konfigurację Adam + Huber Loss.*

---

## D. Zmiana atrybutu predykcji z „High” na „Close”

Zmieniono kolumnę docelową z dziennej ceny maksymalnej (`High` - indeks 1) na cenę zamknięcia sesji (`Close` - indeks 3), modyfikując wywołania wycinania danych na `.iloc[:, 3:4]`.

**Obserwacje:** Prognozowanie cen zamknięcia ma nieco inną dynamikę. Ceny te są wypadkową działań inwestorów na koniec dnia sesyjnego, co generuje inną strukturę trendu niż w przypadku dziennych maksimów. Sieć poprawnie zaadaptowała się do nowego celu, utrzymując stabilność, jednak potwierdziło to odmienną specyfikację tych dwóch atrybutów.

![Wykres predykcji dla Close](3_4_Close%20(1).png)
*Rys 7. Wynik predykcji dla prognozowania cen zamknięcia (Close).*

---

## E. Testowanie mechanizmu Early Stopping i Walidacji

Aby ustrzec model przed przeuczeniem przy zadanej liczbie 100 epok, zastosowano callback `EarlyStopping`. Równocześnie, do rzetelnej oceny zdolności uogólniania sieci, wydzielono 10% danych treningowych jako zbiór walidacyjny (`validation_split=0.1`). Proces monitorowania oparto na metryce `val_loss`.

**Obserwacje:** Użycie tej konfiguracji zapobiegło zjawisku overfittingu. Trening był automatycznie przerywany, gdy błąd na zbiorze walidacyjnym przestawał maleć, co oszczędziło zasoby obliczeniowe i zapobiegło memorizacji danych uczących.

![Wykres Early Stopping](val_loss_earlystop.png)
*Rys 8. Działanie mechanizmu Early Stopping bazującego na metryce val_loss.*

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
Zastosowanie optymalizatora adaptacyjnego (Adam) w połączeniu z funkcją błędu uodpornioną na anomalie rynkowe (Huber) doprowadziło do znacznej redukcji błędu średniokwadratowego, spełniając tym samym kryterium osiągnięcia RMSE poniżej wartości 2.0. Wdrożenie podziału walidacyjnego wraz z mechanizmem Early Stopping zapewniło, że nauczony model posiada faktyczną zdolność do generalizacji trendów, zamiast jedynie odtwarzać dane z próby treningowej.
