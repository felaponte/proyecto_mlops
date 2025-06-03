from locust import HttpUser, task, between

class UsuarioDeCarga(HttpUser):
    wait_time = between(1, 2.5)

    @task
    def hacer_inferencia(self):
        payload = {
                "age": 15,
                "admission_type_id": 1,
                "discharge_disposition_id": 1,
                "admission_source_id": 7,
                "time_in_hospital": 3,
                "num_lab_procedures": 1,
                "num_procedures": 0,
                "num_medications": 13,
                "number_diagnoses": 9,
                "diabetesMed": 1
        }
        # Enviar una petición POST al endpoint /predict
        response = self.client.post("/predict?modelo_elegir=best_gradient_boosting", json=payload)
        # Opcional: validación de respuesta
        if response.status_code != 200:
            print("❌ Error en la inferencia:", response.text)
