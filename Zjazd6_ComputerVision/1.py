import cv2
import mediapipe as mp
"""
    Autorzy:
        Kamil Powierza
        Dawid Feister

    Wymagane biblioteki:
        1. cv2
        2. mediapipe

    Description:
        Program detekcji twarzy i rąk w czasie rzeczywistym, który wykorzystuje bibliotekę MediaPipe do analizy obrazu wideo. 
        Program wykrywa twarze i ręce w obrazie oraz generuje odpowiednie reakcje na podstawie pozycji rąk. 
        Funkcje obejmują wykrywanie podniesionych rąk (wyświetlenie napisu "Poddanie się!") oraz rysowanie celownika na twarzy,
        gdy ręce są opuszczone i wykryty zostaje ruch.
"""
"""
    Description: 
        Inicjalizacja bibliotek MediaPipe do rysowania, detekcji rąk i twarzy.
        Biblioteka `drawing_utils` umożliwia rysowanie punktów i linii na obrazach,
        `hands` służy do wykrywania i śledzenia pozycji rąk, a `face_detection` do wykrywania twarzy.
"""
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection


def draw_crosshair(image, x, y, size=10, color=(0, 0, 255), thickness=2):
    """
        Rysuje celownik na obrazie w zadanej lokalizacji.

        Args:
            image (ndarray): Obraz, na którym ma zostać narysowany celownik.
            x (int): Współrzędna x punktu centralnego celownika.
            y (int): Współrzędna y punktu centralnego celownika.
            size (int, optional): Rozmiar celownika. Domyślnie 10.
            color (tuple, optional): Kolor celownika w formacie RGB. Domyślnie czerwony (0, 0, 255).
            thickness (int, optional): Grubość linii celownika. Domyślnie 2.

        Returns:
            None
    """
    cv2.circle(image, (x, y), size - 3, color, thickness)
    cv2.line(image, (x - size, y), (x + size, y), color, thickness)
    cv2.line(image, (x, y - size), (x, y + size), color, thickness)

"""
    Description: 
        Inicjalizacja przechwytywania obrazu z kamery, gdzie `cv2.VideoCapture(0)` otwiera domyślny strumień kamery.
        Strumień wideo jest używany do analizy obrazu w czasie rzeczywistym.
"""
cap = cv2.VideoCapture(0)

"""
    Description: 
        Zmienne przechowujące:
            previous_frame - Poprzedni obraz, wykorzystywany do analizy ruchu.
            motion_threshold - Próg detekcji ruchu (jak duża musi być zmiana, aby uznać ją za ruch).
            motion_area_threshold - Próg minimalnej powierzchni, aby wykryty ruch był uznany za istotny.
"""
previous_frame = None
motion_threshold = 20
motion_area_threshold = 200


with mp_hands.Hands(
    model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5
) as hands, mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    """
        Pętla główna programu, która analizuje obraz wideo z kamery, wykrywa twarze i ręce, 
        a także reaguje na wykryty ruch oraz pozycje rąk.

        Args:
            hands (mp_hands.Hands): Obiekt do wykrywania i śledzenia rąk.
            face_detection (mp_face_detection.FaceDetection): Obiekt do wykrywania twarzy.

        Returns:
            None
    """
    while cap.isOpened():
        """
            Description: 
                Odczyt obrazu z kamery.
                Jeśli obraz nie zostanie poprawnie odczytany, pętla zostanie przerwana.
        """
        ret, frame = cap.read()
        if not ret:
            print("Nie można odczytać strumienia wideo")
            break

        """
            Description: 
                Konwersja obrazu do dwóch formatów:
                - Odcieni szarości (dla analizy ruchu).
                - RGB (do analizy za pomocą MediaPipe).
        """
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        """
            Description: 
                Przetwarzanie obrazu za pomocą detektorów twarzy i rąk w MediaPipe.
        """
        results_face = face_detection.process(frame_rgb)
        results_hands = hands.process(frame_rgb)

        """
            Description: 
                Obliczanie wymiarów obrazu w celu późniejszego obliczenia pozycji celownika.
        """
        height, width, _ = frame.shape

        """
            Description: 
                Jeśli poprzedni obraz jest nieznany (pierwszy obrót pętli), przechowujemy bieżący obraz.
        """
        if previous_frame is None:
            previous_frame = frame_gray
            continue

        """
            Description: 
                Obliczanie różnicy między bieżącym a poprzednim obrazem w celu wykrycia ruchu.
                Tworzenie maski ruchu na podstawie tej różnicy.
        """
        frame_diff = cv2.absdiff(previous_frame, frame_gray)
        _, motion_mask = cv2.threshold(frame_diff, motion_threshold, 255, cv2.THRESH_BINARY)
        motion_mask = cv2.dilate(motion_mask, None, iterations=2)
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        """
            Description: 
                Sprawdzanie, czy wykryto ruch na podstawie obszaru zmiany w masce.
        """
        motion_detected = any(cv2.contourArea(contour) > motion_area_threshold for contour in contours)

        surrender = False
        if results_face.detections:
            """
                Description: 
                    Iteracja po wykrytych twarzach w obrazie.
                    Dla każdej twarzy obliczane są współrzędne prostokąta obejmującego twarz.
            """
            for detection in results_face.detections:
                """
                    Description:
                        Przetwarzanie wyników detekcji twarzy, aby obliczyć współrzędne prostokąta otaczającego twarz na obrazie.

                    Args:
                        results_face.detections (list): Lista obiektów detekcji twarzy, zawierająca dane na temat wykrytych twarzy.

                    Variables:
                        bboxC (BoundingBox): Obiekt zawierający współrzędne względne prostokąta otaczającego twarz.
                        x_min (int): Współrzędna x lewego górnego rogu prostokąta otaczającego twarz, przekształcona do pikseli.
                        y_min (int): Współrzędna y lewego górnego rogu prostokąta otaczającego twarz, przekształcona do pikseli.
                        bbox_width (int): Szerokość prostokąta otaczającego twarz, przekształcona do pikseli.
                        bbox_height (int): Wysokość prostokąta otaczającego twarz, przekształcona do pikseli.

                    Returns:
                        None
                """
                bboxC = detection.location_data.relative_bounding_box
                x_min = int(bboxC.xmin * width)
                y_min = int(bboxC.ymin * height)
                bbox_width = int(bboxC.width * width)
                bbox_height = int(bboxC.height * height)

                """
                    Description: 
                        Obliczanie współrzędnych do narysowania celownika na twarzy.
                """
                crosshair_x = x_min + bbox_width // 2
                crosshair_y = y_min + bbox_height // 64

                if results_hands.multi_hand_landmarks:
                    """
                        Description: 
                            Iteracja po wykrytych rękach w obrazie.
                    """
                    for hand_landmarks in results_hands.multi_hand_landmarks:
                        """
                            Description:
                                Przetwarzanie wyników detekcji rąk, aby obliczyć pozycję nadgarstka i sprawdzić, czy znajduje się on nad twarzą.

                            Args:
                                results_hands.multi_hand_landmarks (list): Lista obiektów zawierających współrzędne punktów charakterystycznych dla wykrytych rąk.

                            Variables:
                                wrist (Landmark): Obiekt reprezentujący punkt nadgarstka na ręce.
                                wrist_y (float): Współrzędna y nadgarstka w pikselach, przekształcona na podstawie wysokości obrazu.
                                face_bbox (BoundingBox): Obiekt zawierający współrzędne względne prostokąta otaczającego twarz.
                                face_top (int): Współrzędna y górnej krawędzi prostokąta otaczającego twarz, przekształcona na wartość w pikselach.

                            Returns:
                                None
                        """
                        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                        wrist_y = wrist.y * height
                        face_bbox = results_face.detections[0].location_data.relative_bounding_box
                        face_top = int(face_bbox.ymin * height)

                        """
                            Description: 
                                Sprawdzanie, czy ręka znajduje się nad twarzą.
                                Jeśli tak, oznacza to poddanie się.
                        """
                        if wrist_y < face_top:
                            surrender = True
                            break

                if not surrender and motion_detected:
                    """
                        Description: 
                            Rysowanie celownika na twarzy, jeśli wykryto ruch i ręce są opuszczone.
                    """
                    draw_crosshair(frame, crosshair_x, crosshair_y)

        if surrender:
            """
                Description: 
                    Jeśli ręka znajduje się nad twarzą, wyświetlenie napisu "Poddanie się!" na ekranie.
            """
            cv2.putText(frame, "Poddanie sie!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        """
            Description: 
                Ustawienie poprzedniego obrazu na bieżący obraz do analizy w kolejnej iteracji.
        """
        previous_frame = frame_gray

        """
            Description: 
                Wyświetlanie obrazu w czasie rzeczywistym z detekcją twarzy i rąk.
        """
        cv2.imshow("Real-time Detection", frame)

        """
            Description: 
                Zakończenie programu po naciśnięciu klawisza 'ESC'.
        """
        if cv2.waitKey(5) & 0xFF == 27:
            break

"""
    Description: 
        Zwolnienie zasobów kamery i zamknięcie wszystkich okien wideo po zakończeniu programu.
"""
cap.release()
cv2.destroyAllWindows()