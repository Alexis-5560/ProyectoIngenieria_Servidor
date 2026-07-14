import os
import io
import json
import emoji
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from pydantic import BaseModel
import google.generativeai as genai
from contextlib import asynccontextmanager
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
import shap


#########-------       uvicorn main:app --host 0.0.0.0 --port 8000 --reload        -----------#############  
#uvicorn main:app --reload 


genai.configure(api_key="AQ.Ab8RN6LBoVtTT7or2VDLvDFh4XSdZ138MKrqMvhLJD36w9t7Hg") 

# Ruta del modelo entrenado
RUTA_MODELO = r"C:\Users\usuario\Desktop\servidor_ia_violencia\mi_modelo_final_acoso1"


TEMPERATURA_CALIBRADA = 1.9921

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Iniciando carga del modelo RoBERTa y SHAP en memoria...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(RUTA_MODELO)
        modelo = AutoModelForSequenceClassification.from_pretrained(RUTA_MODELO)
        modelo.eval()  # modo evaluación: desactiva dropout y otras capas de entrenamiento

        ml_models["tokenizer"] = tokenizer
        ml_models["modelo"] = modelo
        ml_models["temperatura"] = TEMPERATURA_CALIBRADA

        # SHAP necesita la interfaz de un pipeline, así que se conserva solo para esto.
        # No se usa para la predicción final mostrada al usuario.
        clasificador_pipeline = pipeline(
            task="text-classification",
            model=modelo,
            tokenizer=tokenizer,
            top_k=None
        )
        ml_models["explainer"] = shap.Explainer(clasificador_pipeline)

        print("Modelo RoBERTa y explicador SHAP cargados exitosamente.")
        print(f"Temperatura de calibración activa: {TEMPERATURA_CALIBRADA}")
    except Exception as e:
        print(f"Error crítico al cargar la IA: {e}")
    
    yield
    ml_models.clear()
    print("Memoria RAM liberada correctamente.")

app = FastAPI(title="Backend Violencia Digital", lifespan=lifespan)


def predecir_con_temperatura(texto: str) -> dict:
    """
    Predicción final usando el modelo y tokenizador directamente (sin pipeline),
    aplicando Temperature Scaling para que el score de confianza sea más honesto
    y no esté artificialmente cercano al 100% en casos no tan claros.
    """
    tokenizer = ml_models["tokenizer"]
    modelo = ml_models["modelo"]
    temperatura = ml_models["temperatura"]

    inputs = tokenizer(texto, return_tensors="pt", truncation=True, max_length=128)

    with torch.no_grad():
        logits = modelo(**inputs).logits

    probs = F.softmax(logits / temperatura, dim=-1)
    confianza, clase_idx = torch.max(probs, dim=-1)

    etiqueta = modelo.config.id2label[clase_idx.item()]

    return {
        "label": etiqueta,
        "score": confianza.item()
    }


def resultadosSHAP(texto_preparado: str, resultado_final: dict):
    """Esta función correrá de forma invisible sin trabar la app móvil"""
    try:
        explicacion = ml_models["explainer"]([texto_preparado])
        palabras_dict = {}

        for i, token in enumerate(explicacion[0].data):
            peso_fragmento = max(abs(val) for val in explicacion[0].values[i])
            token_limpio = token.replace('Ġ', '').replace('##', '').strip(" .,!?¿¡\n\t\"'")
            if len(token_limpio) > 2 and token_limpio.isalpha():
                palabras_dict[token_limpio] = max(palabras_dict.get(token_limpio, 0.0), float(peso_fragmento))

        todos_los_tokens = [{"palabra": p, "peso": round(w, 4)} for p, w in palabras_dict.items()]
        
        top_tokens = sorted(todos_los_tokens, key=lambda x: x["peso"], reverse=True)[:15]

        #Guarado
        fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        registro = f"[{fecha_actual}]\nTexto extraido de Gemini:\n{texto_preparado}\nResultado: {resultado_final['label']}\n"
        if top_tokens:
            for t in top_tokens:
                registro += f"  - {t['palabra']}: {t['peso']}\n"
        registro += "FIN DE PRUEBA\n"

        ruta_txt = r"C:\Users\usuario\Desktop\servidor_ia_violencia\historial_evaluaciones.txt"
        with open(ruta_txt, "a", encoding="utf-8") as archivo_log:
            archivo_log.write(registro)
            
        print(f"Archivo Actualizado")

    except Exception as e:
        print(f"Error en resultadosSHAP: {e}")

def demojizar_texto(texto: str) -> str:
    return emoji.demojize(texto, language='es', delimiters=("[emoji_", "]"))


@app.post("/analizar-chat")
async def analizar_chat(tareas_fondo: BackgroundTasks, imagen: UploadFile = File(...)):
    try:
        contenido = await imagen.read()
        from PIL import Image
        img = Image.open(io.BytesIO(contenido))

        # Gemini
        modelo_vision = genai.GenerativeModel('gemini-2.5-flash')
        instrucciones = (
            "Eres un experto en OCR. Extrae SOLO el contenido de los mensajes de esta conversación. "
            "Ignora horas, fechas, nombres y UI. Conserva TODOS los emojis originales. "
            "Devuelve ÚNICAMENTE un arreglo JSON válido donde cada elemento sea un mensaje (string)."
        )
        
        respuesta = modelo_vision.generate_content([instrucciones, img])
        texto_ia = respuesta.text

        if texto_ia.startswith("```json"):
            texto_ia = texto_ia.replace("```json", "").replace("```", "").strip()
            
        try:
            lista_mensajes = json.loads(texto_ia)
            conversacion_unida = " ".join(lista_mensajes)
        except:
            conversacion_unida = texto_ia

        texto_preparado = demojizar_texto(conversacion_unida)

        # Análisis con RoBERTa (predicción calibrada con Temperature Scaling)
        if "modelo" not in ml_models:
            raise RuntimeError("El modelo no está disponible.")

        resultado_final = predecir_con_temperatura(texto_preparado)

        tareas_fondo.add_task(resultadosSHAP, texto_preparado, resultado_final)

        return {
            "estatus": "Éxito",
            "detalles": {
                "resultado_modelo": resultado_final
            }
        }
    except Exception as e:
        print(f"Error procesando captura: {e}")
        return {"estatus": "Error", "mensaje": str(e)}