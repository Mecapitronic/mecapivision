import cv2
import time

def list_cameras(max_tested=10):
    print("🔍 Recherche des caméras disponibles...\n")
    available_cams = []

    for i in range(max_tested):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # CAP_DSHOW recommandé sous Windows
        if cap.isOpened():
            print(f"✅ Caméra trouvée à l'index {i}")
            available_cams.append(i)
            cap.release()
        else:
            cap.release()

    if not available_cams:
        print("❌ Aucune caméra détectée.")
    return available_cams


def test_camera(index, duration=5):
    print(f"\n🎥 Test de la caméra {index} pendant {duration} secondes...")
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"❌ Impossible d'ouvrir la caméra {index}")
        return

    start_time = time.time()

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            print("❌ Erreur de lecture d'image")
            break

        cv2.imshow(f"Camera {index}", frame)

        # Quitter si on appuie sur 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Fin du test caméra {index}")


if __name__ == "__main__":
    cams = list_cameras()

    for cam_index in cams:
        test_camera(cam_index, duration=5)

    print("\n🎉 Test terminé.")
