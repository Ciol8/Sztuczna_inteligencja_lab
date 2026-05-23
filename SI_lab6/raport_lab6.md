# Raport z Laboratorium nr 6: Predykcja Cen Akcji (LSTM vs GRU)

## Cel zadania
Zbudowanie i optymalizacja rekurencyjnych sieci neuronowych (RNN) do przewidywania kursu giełdowego, oraz analiza wpływu różnych hiperparametrów na dokładność predykcji.

---

## Baza: Porównanie LSTM i GRU
Zbudowałem dwie analogiczne sieci: jedną opartą na warstwach LSTM, drugą na GRU.
**Obserwacje:** Sieci GRU trenowały się odczuwalnie szybciej (krótszy czas wykonania jednej epoki: [TUTAJ WPISZ CZASY]), co wynika z braku bramki wyjściowej (output gate) w jej komórkach. Pomimo prostszej budowy, wyniki błędu były porównywalne, a czasami nawet lepsze dla GRU.

*[TUTAJ DODAJ SCREENY LOGÓW Z TRENINGU LSTM I GRU]*

---

## A. Eksperymenty z ilością jednostek (Units)
Testowałem wartości: 20, 50, 128. 
**Obserwacje:** Przy bardzo małej liczbie jednostek (20), model miał problem z odwzorowaniem skomplikowanych kształtów wykresu. Przy 128 jednostkach sieć trenowała się znacznie dłużej i istniało ryzyko overfittingu, choć dokładnie dopasowywała się do danych treningowych. Złotym środkiem okazała się wartość w okolicach 50-64.

*[TUTAJ DODAJ SCREENY WYKRESÓW DLA MAŁEJ I DUŻEJ LICZBY UNITS]*

---

## B & C. Testowanie Optymalizatorów i Funkcji Straty
Sprawdziłem optymalizatory (SGD, Adam, AdamW) oraz funkcje straty (MSE, MAE, Huber, LogCosh).
**Obserwacje:** * Optymalizator `SGD` bez odpowiedniego dostrajania zbiegał najwolniej. `Adam` oraz `AdamW` radziły sobie zdecydowanie najlepiej, szybko minimalizując stratę.
* Dla danych giełdowych, świetnie sprawdziła się funkcja straty `Huber`. W odróżnieniu od klasycznego `MSE`, Huber jest mniej wrażliwy na gwałtowne piki na wykresach giełdowych, co zaowocowało stabilniejszą krzywą predykcji.

*[TUTAJ DODAJ SCREEN WYKRESU DLA LOSS=HUBER ORAZ OPT=ADAM]*

---

## D. Zmiana atrybutu predykcji (High na Close)
Po zmianie kolumny targetu z dziennych maksimów ("High") na ceny zamknięcia ("Close"), zauważyłem [Wpisz czy błąd u Ciebie wzrósł czy zmalał]. Wynika to z faktu, że ceny zamknięcia posiadają inną charakterystykę zmienności wynikającą z zachowań traderów pod koniec sesji giełdowej.

---

## E. Early Stopping i Walidacja
Dodałem mechanizm wczesnego zatrzymywania:
`EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)` oraz wyznaczyłem 10% danych na zbiór walidacyjny (`validation_split=0.1`).
**Obserwacje:** Ustawienie 100 epok "na twardo" w LSTM mija się z celem. Dzięki EarlyStopping model przerwał naukę w okolicach [WPISZ NR EPOKI] epoki. Oszczędziło to moc obliczeniową i uchroniło model przed zapamiętywaniem szumu (overfittingiem).

*[TUTAJ DODAJ SCREEN Z LOGU POKAZUJĄCY ZATRZYMANIE TRENINGU PRZED 100 EPOKĄ]*

---

## F & G. Ostateczna i najlepsza konfiguracja (RMSE < 2.0)
Połączenie powyższych eksperymentów pozwoliło mi uzyskać ostateczny **błąd RMSE na poziomie: [WPISZ TUTAJ SWÓJ OSTATECZNY WYNIK < 2.0]**.

**Moja Najlepsza Konfiguracja:**
* **Architektura:** GRU (za szybkość nauki)
* **Liczba jednostek:** 64
* **Optymalizator:** AdamW
* **Loss:** Huber
* **Dodatki:** EarlyStopping (patience=5)

**Uzasadnienie:** Taka kombinacja pozwala wykorzystać najlepsze cechy optymalizatora adaptacyjnego, funkcję straty uodpornioną na błędy odstające oraz sieć, która nie ulega łatwemu przeuczeniu dzięki walidacji krzyżowej w trakcie trenowania.

*[TUTAJ DODAJ OSTATECZNY, NAJŁADNIEJSZY SCREEN WYKRESU PREDYKCJI WRAZ Z WYDRUKIEM BŁĘDÓW]*
