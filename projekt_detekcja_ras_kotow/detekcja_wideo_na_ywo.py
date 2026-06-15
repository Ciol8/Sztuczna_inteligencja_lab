# ==============================================================================
# Detekcja ras kotów na strumieniu wideo (OpenCV + YOLOv8)
# ==============================================================================

# KROK 1: Wymagane biblioteki
import cv2
import time
from ultralytics import YOLO

# ==============================================================================
# KROK 2: INICJALIZACJA MODELU I WIDEO
# ==============================================================================
print("Wczytywanie wyuczonego modelu...")
# Podajemy ścieżkę do naszego wytrenowanego pliku z wagami
model = YOLO('best_n.pt')

print("Otwieranie pliku wideo...")
# Wczytujemy film z dysku. 
# Aby podłączyć kamerę internetową, wystarczy zmienić
# 'koty_film.mp4' na cyfrę 0, np.: cap = cv2.VideoCapture(0)
sciezka_do_wideo = 'koty_film.mp4' 
cap = cv2.VideoCapture(sciezka_do_wideo)

# Sprawdzamy, czy wideo załadowało się poprawnie
if not cap.isOpened():
    print("Błąd: Nie można otworzyć pliku wideo!")
    exit()

# ==============================================================================
# KROK 3: GŁÓWNA PĘTLA - ANALIZA KLATKA PO KLATCE
# ==============================================================================
print("Rozpoczynam analizę wideo. Wciśnij klawisz 'q', aby wyjść.")

# Ta pętla będzie kręcić się tak długo, jak długo trwa film
while cap.isOpened():
    # Pobieramy czas przed analizą klatki żeby potem policzyć FPS
    start_time = time.time()
    
    # Odczytujemy jedną pojedynczą klatkę z filmu
    # 'sukces' to zmienna True/False mówiąca, czy klatka została odczytana,
    # 'klatka' to sam obraz w postaci macierzy pikseli
    sukces, klatka = cap.read()
    
    # Jeśli sukces = False, oznacza to, że film się skończył
    if not sukces:
        print("Koniec wideo.")
        break

    # --- INFERENCJA  ---
    # Wysyłamy naszą klatkę do modelu YOLO. Parametr verbose=False wycisza logi w konsoli, 
    # a conf=0.5 pokazuje tylko pewne predykcje.
    wyniki = model(klatka, conf=0.5, verbose=False)
    
    # YOLO posiada wbudowaną, która automatycznie
    # bierze naszą oryginalną klatkę i rysuje na niej prostokąty (Bounding Boxes) oraz nazwy ras.
    klatka_z_ramkami = wyniki[0].plot()

    # --- LICZENIE PRĘDKOŚCI FPS ---
    # Pobieramy czas po zakończeniu rysowania i myślenia modelu
    end_time = time.time()
    
    # Czas przetwarzania tej jednej klatki
    czas_przetwarzania = end_time - start_time
    # Obliczamy ile takich klatek zmieści się w jednej sekundzie (1 / czas)
    fps = 1 / czas_przetwarzania
    
    # Nakładamy tekst z informacją o FPS w lewym górnym rogu ekranu
    # cv2.putText(na czym rysujemy, tekst, (x,y), czcionka, wielkość, (kolor BGR), grubość)
    cv2.putText(klatka_z_ramkami, f'FPS: {int(fps)}', (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # --- WYSWIETLANIE NA EKRANIE ---
    # Otwieramy okno systemowe z naszym filmem
    cv2.imshow('Detektor Ras Kotow', klatka_z_ramkami)

    # Zabezpieczenie: Jeśli użytkownik wciśnie klawisz 'q', przerywamy pętlę
    # (Metoda cv2.waitKey(1) odczekuje 1 milisekundę na reakcję z klawiatury)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Przerwano przez użytkownika.")
        break

# ==============================================================================
# KROK 4: SPRZĄTANIE PAMIĘCI
# ==============================================================================
# Zwalniamy plik wideo z pamięci RAM i zamykamy wszystkie okienka Windows/Mac
cap.release()
cv2.destroyAllWindows()