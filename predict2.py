import cv2 as cv
import tensorflow as tf
import numpy as np
import time

def preprocess_frame(frame):

    # bgr to rgb
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    
    # resize e pre processamento
    img = tf.image.resize(rgb_frame, (224, 224))
    img = tf.cast(img, tf.float32) / 255.0
    
    # add batch dimension
    img = tf.expand_dims(img, 0)
    return img

def main():
    # carregar o modelo
    print("Loading model...")
    model = tf.keras.models.load_model('models/model.keras')
    print("Model loaded successfully!")

    # Inicializar webcam
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # temperatura padrao
    default_temp = tf.expand_dims((25.0 - 20.0) / (100.0 - 20.0), 0)

    print("Starting real-time detection. Press 'q' to quit.")
    

    prev_time = time.time()
    fps = 0
    
    while True:
        # webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # pre processamento do frame
        processed_frame = preprocess_frame(frame)
        
        # predict
        prediction = model.predict([processed_frame, tf.expand_dims(default_temp, 0)], verbose=0)
        confidence = float(prediction[0][0])
        is_fire = confidence >= 0.5

    
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time


        text_color = (0, 0, 255) if is_fire else (0, 255, 0)  # Red if fire, Green if no fire
        cv.putText(frame, f"Fire: {is_fire} ({confidence:.2%})", (10, 30),
                  cv.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        cv.putText(frame, f"FPS: {fps:.1f}", (10, 70),
                  cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


        cv.imshow('Fire Detection', frame)

 
        if cv.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
