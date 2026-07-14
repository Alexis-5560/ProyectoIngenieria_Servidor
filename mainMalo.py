import os
import io
import json
import emoji
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from pydantic import BaseModel
import google.generativeai as genai
from contextlib import asynccontextmanager
from transformers import pipeline
import shap

#########-------       uvicorn main:app --host 0.0.0.0 --port 8000 --reload        -----------#############  
#uvicorn main:app --reload 

genai.configure(api_key="AQ.Ab8RN6JTkBWNtfmHATHrC4Mlg9J4TwIkoq6yDeNBUY3MTR1geA") 

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Iniciando carga del modelo RoBERTa y SHAP en memoria...")
    try:
        clasificador = pipeline(
            task="text-classification",
            model=r"C:\Users\usuario\Desktop\servidor_ia_violencia\mi_modelo_final_acoso1",
            tokenizer=r"C:\Users\usuario\Desktop\servidor_ia_violencia\mi_modelo_final_acoso1",
            top_k=None 
        )
        ml_models["clasificador_violencia"] = clasificador
        ml_models["explainer"] = shap.Explainer(clasificador)
        print("Modelo RoBERTa y explicador SHAP cargados exitosamente.")
    except Exception as e:
        print(f"Error crítico al cargar la IA: {e}")
    
    yield
    ml_models.clear()
    print("Memoria RAM liberada correctamente.")

app = FastAPI(title="Backend Violencia Digital", lifespan=lifespan)


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
        from datetime import datetime
        fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        registro = f"[{fecha_actual}]\nTexto extraido de Gemini:\n{texto_preparado}\Resultado: {resultado_final['label']}\n"
        if top_tokens:
            for t in top_tokens:
                registro += f"  - {t['palabra']}: {t['peso']}\n"
        registro += "FIN DE PRUEBA"

        import os
        ruta_txt = r"C:\Users\usuario\Desktop\servidor_ia_violencia\historial_evaluaciones.txt"
        with open(ruta_txt, "a", encoding="utf-8") as archivo_log:
            archivo_log.write(registro)
            
        print(f"Archivo Actualizado")

    except Exception as e:
        print(f"Error")

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

        # Analisis de roberta
        if "clasificador_violencia" not in ml_models:
            raise RuntimeError("El modelo no está disponible.")
            
        prediccion_cruda = ml_models["clasificador_violencia"](texto_preparado)
        resultado_final = max(prediccion_cruda[0], key=lambda x: x['score'])

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

        
    
    except Exception as e:
        print(f"Error procesando captura: {e}")
        return {"estatus": "Error", "mensaje": str(e)}