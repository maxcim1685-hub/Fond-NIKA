import json
import os
import re
import webbrowser
from bs4 import BeautifulSoup
from catboost import CatBoostClassifier
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import requests
import uvicorn
import subprocess
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from io import BytesIO
from PIL import Image
import torch
from torchvision import models, transforms

# Загружаем предобученную нейросеть MobileNetV2 для распознавания фото
print("⏳ Загрузка ИИ-модели компьютерного зрения...")
weights = models.MobileNet_V2_Weights.DEFAULT
vision_model = models.mobilenet_v2(weights=weights)
vision_model.eval() # Переводим в режим предсказания

# Подготовка картинки под стандарт нейросети
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
print("✅ ИИ-помощник для распознавания кошек и собак готов!")


def classify_pet_by_photo(image_url: str) -> str:
  """Скачивает фото по ссылке и определяет через нейросеть: 'cat', 'dog' или 'all'."""
  if not image_url:
    return "all"
  try:
    resp = requests.get(
        image_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5
    )
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0)

    with torch.no_grad():
      output = vision_model(input_tensor)
      # Получаем индекс самого вероятного класса (от 0 до 999)
      predicted_idx = torch.argmax(output[0]).item()

    # В стандартной разметке ImageNet:
    # 151..268 — все породы собак
    # 281..285 — кошки (табби, тигровая, персидская, сиамская, египетская)
    if 281 <= predicted_idx <= 285:
      return "cat"
    elif 151 <= predicted_idx <= 268:
      return "dog"
  except Exception as e:
    print(f"Ошибка анализа фото {image_url}:", e)
  return "all"

app = FastAPI()

USERS_FILE = "users.json"
FAVORITES_FILE = "favorites.json"
PETS_CACHE_FILE = "pets_cache.json"
ADMIN_PASSWORD = "admin2026"

cb_model = CatBoostClassifier()
model_loaded = False

for model_file in ["catboost_model.cbm", "catboost_model.bin"]:
  if os.path.exists(model_file):
    try:
      cb_model.load_model(model_file)
      model_loaded = True
      print(f"Модель CatBoost загружена: {model_file}")
      break
    except Exception as e:
      print(f"Ошибка загрузки {model_file}: {e}")

if not model_loaded:
  cb_model = None
  print("Файл модели не найден, включен резервный расчёт вероятности.")
class UserReg(BaseModel):
  first_name: str
  last_name: str
  email: str


class SurveyRequest(BaseModel):
    email: str
    # 9 вопросов с выбором
    housing_type: str          # "квартира" / "частный дом"
    housing_ownership: str     # "съемная" / "свое"
    landlord_permission: str   # "да" / "нет"
    spouse_consent: str        # "да" / "нет"
    has_children: str          # "да" / "нет"
    has_pets: str              # "да" / "нет"
    has_experience: str        # "да" / "нет"
    considers_guardianship: str  # "да" / "нет"
    considers_adoption: str    # "да" / "нет"
    # 2 открытых вопроса
    experience_text: str       # описание опыта
    motivation_text: str       # мотивация


class PetAction(BaseModel):
  user_email: str
  id: str
  name: str
  age: str
  image: str
  url: str
  species: str


class RemoveAction(BaseModel):
  user_email: str
  id: str


class AdminAuth(BaseModel):
  password: str


def load_json(filename, default):
  if not os.path.exists(filename):
    return default
  try:
    with open(filename, "r", encoding="utf-8") as f:
      return json.load(f)
  except Exception:
    return default


def save_json(filename, data):
  with open(filename, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)


def is_valid_email(email: str) -> bool:
  pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
  return bool(re.match(pattern, email))


def fetch_html_safe(url: str) -> str:
  
  headers = {
      "User-Agent": (
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
          " (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
      ),
      "Referer": "https://fond-nika.ru",
  }
  try:
    resp = requests.get(url, headers=headers, timeout=8)
    return resp.text
  except Exception:
    
    cmd = ["curl", "-s", "-L", "-A", headers["User-Agent"], url]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    return res.stdout


from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By


def get_first_card_name(driver) -> str:
  """Возвращает имя первого питомца на текущей странице для контроля обновления фильтра."""
  try:
    el = driver.find_element(By.CSS_SELECTOR, "[class*='item-title']")
    return el.text.strip()
  except Exception:
    return ""


def switch_nika_category(driver, category_name: str):
  old_first_name = get_first_card_name(driver)

  # Ищем ссылки/кнопки/лейблы, содержащие слово Кошки или Собаки (с учетом вложенных тегов через точку)
  elements = driver.find_elements(
      By.XPATH,
      f"//a[contains(., '{category_name}')] | //label[contains(.,"
      f" '{category_name}')] | //button[contains(., '{category_name}')]",
  )
  clicked = False

  for el in elements:
    txt = el.text.strip()
    # Берем только саму короткую кнопку фильтра (а не длинную ссылку в статье)
    if txt and len(txt) < 25:
      driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", el)
      time.sleep(0.5)
      driver.execute_script("arguments[0].click();", el)
      clicked = True
      print(f"✅ Нажата кнопка '{txt}' (тег: {el.tag_name})")
      break

  if not clicked:
    print(f"⚠️ Вкладка '{category_name}' не найдена на экране.")
    return

  # Ждем до 8 секунд, пока на странице сменится первая карточка
  for _ in range(16):
    time.sleep(0.5)
    new_first_name = get_first_card_name(driver)
    if new_first_name and new_first_name != old_first_name:
      break
  time.sleep(1)



def scrape_visible_cards(driver, species_type: str, max_pages: int = 3) -> list:
  """Собирает карточки с текущей активной вкладки и листает страницы пагинации."""
  items = []
  seen_ids = set()

  for page_num in range(1, max_pages + 1):
    time.sleep(1.5)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    title_blocks = soup.find_all(class_=re.compile(r"item-title"))
    if not title_blocks:
      break

    new_on_page = 0
    for title_div in title_blocks:
      name = title_div.get_text(strip=True)

      card = title_div.parent
      for _ in range(5):
        if (
            card
            and card.find(class_=re.compile(r"item-age"))
            and card.find("img")
        ):
          break
        if card.parent:
          card = card.parent

      if not card:
        continue

      age_div = card.find(class_=re.compile(r"item-age"))
      age_text = (
          age_div.get_text(" ", strip=True) if age_div else "Возраст не указан"
      )

      link_tag = title_div.find("a", href=True) or card.find(
          "a", href=re.compile(r"/ourpets/")
      )
      if not link_tag:
        continue

      raw_href = link_tag["href"].strip()
      clean_href, _, _ = raw_href.partition("?")
      pet_url = (
          "https://fond-nika.ru" + clean_href
          if clean_href.startswith("/")
          else clean_href
      )
      pet_id = pet_url.rstrip("/")

      if pet_id in seen_ids:
        continue

      img_tag = card.find("img")
      img_src = ""
      if img_tag:
        img_src = img_tag.get("data-src") or img_tag.get("src") or ""
        if img_src and not img_src.startswith("http"):
          img_src = "https://fond-nika.ru" + img_src

      seen_ids.add(pet_id)
      new_on_page += 1
      items.append({
          "id": pet_id,
          "name": name,
          "age": age_text,
          "image": img_src,
          "url": pet_id + "/",
          "species": species_type,
      })

    if new_on_page == 0 or page_num == max_pages:
      break

    # Переходим на следующую страницу (цифра 2, 3...)
    old_first = get_first_card_name(driver)
    try:
      next_btns = driver.find_elements(
          By.XPATH, f"//a[normalize-space(text())='{page_num + 1}']"
      )
      for btn in next_btns:
        if btn.is_displayed():
          driver.execute_script(
              "arguments[0].scrollIntoView({block: 'center'});", btn
          )
          time.sleep(0.3)
          try:
            btn.click()
          except Exception:
            driver.execute_script("arguments[0].click();", btn)
          break

      # Ждем смены карточек после перехода на следующую страницу
      for _ in range(12):
        time.sleep(0.5)
        if get_first_card_name(driver) != old_first:
          break
    except Exception:
      break

  return items


def detect_species_fast(pet_url: str) -> str:
  """Определяет вид питомца по характерным словам в его личной анкете."""
  try:
    r = requests.get(
        pet_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5
    )
    soup = BeautifulSoup(r.text, "html.parser")

    # Удаляем шапку, меню и подвал, чтобы не мешала фраза "приют для собак и кошек"
    for tag in soup.find_all(["header", "footer", "nav"]):
      tag.decompose()

    text = soup.get_text(" ").lower()
    text = text.replace("для собак и кошек", "").replace("собак и кошек", "")

    cat_words = (
        "кошка",
        "кошки",
        "кошку",
        "кот ",
        "кота",
        "коту",
        "котик",
        "котёнок",
        "котенок",
        "лоток",
        "лотку",
        "лотка",
        "когтеточк",
        "мурлы",
    )
    dog_words = (
        "собака",
        "собаки",
        "собаку",
        "пёс",
        "пес",
        "пса",
        "щенок",
        "щенка",
        "поводок",
        "поводке",
        "выгул",
        "вольер",
    )

    cat_score = sum(text.count(w) for w in cat_words)
    dog_score = sum(text.count(w) for w in dog_words)

    if cat_score > dog_score:
      return "cat"
    elif dog_score > cat_score:
      return "dog"
  except Exception:
    pass
  return "dog"


def refresh_pets_database():
  """Собирает карточки с общего каталога НИКА и сортирует их через ИИ по фотографиям."""
  options = Options()
  options.add_argument("--headless=new")
  options.add_argument("--window-size=1920,1080")
  options.add_argument("--disable-gpu")
  options.add_argument("--no-sandbox")

  driver = webdriver.Chrome(options=options)
  raw_pets = []

  try:
    driver.get("https://fond-nika.ru")
    time.sleep(2.5)
    # Собираем 6 страниц общего каталога (около 72 животных)
    raw_pets = scrape_visible_cards(driver, "all", max_pages=6)
  finally:
    driver.quit()

  print(f" Начинаю анализ {len(raw_pets)} фотографий через ИИ...")
  cats = []
  dogs = []
  others = []

  for pet in raw_pets:
    species = classify_pet_by_photo(pet["image"])
    pet["species"] = species
    if species == "cat":
      cats.append(pet)
    elif species == "dog":
      dogs.append(pet)
    else:
      others.append(pet)

  # Чередуем кошек и собак для общей вкладки "Все"
  combined = []
  for i in range(max(len(cats), len(dogs))):
    if i < len(cats):
      combined.append(cats[i])
    if i < len(dogs):
      combined.append(dogs[i])
  combined.extend(others)

  save_json(PETS_CACHE_FILE, combined)
  print(
      f" ИИ завершил сортировку! Кошек: {len(cats)}, Собак: {len(dogs)},"
      f" Не определено: {len(others)}"
  )
  return combined




def calculate_adoption_probability(survey: SurveyRequest) -> int:
    # 11 признаков в том же порядке, что и при обучении
    features = [
        survey.housing_type,
        survey.housing_ownership,
        survey.landlord_permission,
        survey.spouse_consent,
        survey.has_children,
        survey.has_pets,
        survey.has_experience,
        survey.considers_guardianship,
        survey.considers_adoption,
        survey.experience_text,
        survey.motivation_text
    ]
    
    if cb_model is not None:
        try:
            probs = cb_model.predict_proba([features])
            prob_percent = round(float(probs[0][1]) * 100)
            return max(1, min(99, prob_percent))
        except Exception as e:
            print(f"Ошибка инференса CatBoost: {e}")
    
    # Резервный расчёт (если модель не загружена)
    score = 40
    if survey.housing_type == "частный дом": score += 10
    if survey.housing_ownership == "свое": score += 8
    if survey.landlord_permission == "да": score += 7
    if survey.spouse_consent == "да": score += 10
    if survey.has_children == "нет": score += 5
    if survey.has_pets == "да": score += 8
    if survey.has_experience == "да": score += 12
    if survey.considers_guardianship == "да": score += 5
    if survey.considers_adoption == "да": score += 5
    if len(survey.experience_text) > 20: score += 5
    if len(survey.motivation_text) > 20: score += 5
    return min(98, score)


@app.post("/api/register")
def api_register(user: UserReg):
  fname = user.first_name.strip()
  lname = user.last_name.strip()
  email = user.email.strip().lower()

  if len(fname) < 2 or len(lname) < 2:
    return {"status": "error", "message": "Введите корректные имя и фамилию"}
  if not is_valid_email(email):
    return {
        "status": "error",
        "message": "Некорректный адрес почты (пример: user@mail.ru)",
    }

  users = load_json(USERS_FILE, [])
  for u in users:
    if u["email"] == email:
      return {"status": "ok", "user": u}

  new_user = {
      "first_name": fname,
      "last_name": lname,
      "email": email,
      "survey_completed": False,
      "adoption_probability": None,
      "survey_answers": {},
  }
  users.append(new_user)
  save_json(USERS_FILE, users)
  return {"status": "ok", "user": new_user}
@app.post("/api/survey")
def api_survey(data: SurveyRequest):
  users = load_json(USERS_FILE, [])
  email = data.email.strip().lower()
  probability = calculate_adoption_probability(data)

  updated_user = None
  for u in users:
    if u["email"] == email:
      u["survey_completed"] = True
      u["adoption_probability"] = probability
      u["survey_answers"] = data.model_dump()
      updated_user = u
      break

  if not updated_user:
    raise HTTPException(status_code=404, detail="Пользователь не найден")

  save_json(USERS_FILE, users)
  return {"status": "ok", "user": updated_user}


@app.get("/api/pets")
def api_get_pets(species: str = "all"):
  pets = load_json(PETS_CACHE_FILE, [])
  if not pets:
    pets = refresh_pets_database()

  if species == "cat":
    return [p for p in pets if p.get("species") == "cat"]
  elif species == "dog":
    return [p for p in pets if p.get("species") == "dog"]
  return pets


@app.post("/api/pets/refresh")
def api_refresh_pets():
  pets = refresh_pets_database()
  return {"status": "ok", "count": len(pets)}


@app.post("/api/favorite")
def api_add_favorite(pet: PetAction):
  all_favs = load_json(FAVORITES_FILE, {})
  email = pet.user_email.lower()
  user_favs = all_favs.get(email, [])

  if not any(item["id"] == pet.id for item in user_favs):
    item_dict = pet.model_dump()
    del item_dict["user_email"]
    user_favs.append(item_dict)
    all_favs[email] = user_favs
    save_json(FAVORITES_FILE, all_favs)
  return {"status": "ok"}


@app.post("/api/favorite/remove")
def api_remove_favorite(item: RemoveAction):
  all_favs = load_json(FAVORITES_FILE, {})
  email = item.user_email.lower()
  user_favs = all_favs.get(email, [])
  all_favs[email] = [p for p in user_favs if p["id"] != item.id]
  save_json(FAVORITES_FILE, all_favs)
  return {"status": "ok"}


@app.get("/api/favorites")
def api_get_favorites(email: str = ""):
  all_favs = load_json(FAVORITES_FILE, {})
  return all_favs.get(email.lower(), [])


@app.post("/api/admin/users")
def api_admin_users(auth: AdminAuth):
  if auth.password != ADMIN_PASSWORD:
    raise HTTPException(status_code=403, detail="Неверный пароль администратора")
  users = load_json(USERS_FILE, [])
  return {"status": "ok", "users": users}


HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Фонд НИКА — Подбор питомца</title>
    <style>
        * { box-sizing: border-box; font-family: 'Segoe UI', Tahoma, sans-serif; }
        body {
            background-color: #ffffff;
            color: #1b5e20;
            margin: 0;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .box-card {
            background-color: #ffffff;
            border: 2px solid #1b5e20;
            border-radius: 18px;
            padding: 24px;
            width: 100%;
            max-width: 420px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(27, 94, 32, 0.08);
            margin-bottom: 15px;
        }
        .admin-box { max-width: 780px; }
        h2, h3, label, p { color: #1b5e20; }
        .form-group {
            display: flex;
            flex-direction: column;
            text-align: left;
            margin-bottom: 14px;
        }
        .form-group label {
            font-weight: 600;
            font-size: 14px;
            margin-bottom: 5px;
        }
        .input-field {
            background-color: #ffffff;
            color: #1b5e20;
            border: 2px solid #1b5e20;
            border-radius: 10px;
            padding: 10px 12px;
            font-size: 14px;
            outline: none;
        }
        .btn {
            padding: 10px 20px;
            border-radius: 22px;
            border: 2px solid #1b5e20;
            font-size: 14px;
            font-weight: bold;
            cursor: pointer;
            transition: 0.2s;
        }
        .btn-active {
            background-color: #1b5e20;
            color: #ffffff;
        }
        .btn-inactive {
            background-color: #ffffff;
            color: #1b5e20;
        }
        .btn-inactive:hover { background-color: #f1f8e9; }
        .top-nav, .filter-nav {
            display: flex;
            gap: 10px;
            justify-content: center;
            flex-wrap: wrap;
            margin-bottom: 15px;
        }
        .pet-card { transition: transform 0.35s ease, opacity 0.35s ease; }
        .swipe-left { transform: translateX(-140%) rotate(-12deg); opacity: 0; }
        .swipe-right { transform: translateX(140%) rotate(12deg); opacity: 0; }
        .pet-card img {
            width: 100%;
            height: 340px;
            object-fit: cover;
            border-radius: 12px;
            border: 2px solid #1b5e20;
            cursor: pointer;
        }
        .card-actions {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            margin-top: 18px;
        }
        .card-actions .btn { flex: 1; }
        .error-text { color: #c62828; font-size: 13px; min-height: 18px; margin-bottom: 8px; }
        .fav-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid #1b5e20;
            padding: 10px 0;
            text-align: left;
        }
        .fav-row img {
            width: 56px;
            height: 56px;
            object-fit: cover;
            border-radius: 8px;
            border: 1px solid #1b5e20;
            margin-right: 12px;
        }
        .excel-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            font-size: 14px;
            background-color: #ffffff;
        }
        .excel-table th {
            background-color: #1b5e20;
            color: #ffffff;
            border: 1px solid #1b5e20;
            padding: 10px;
            text-align: left;
        }
        .excel-table td {
            background-color: #ffffff;
            color: #1b5e20;
            border: 1px solid #1b5e20;
            padding: 9px 10px;
            text-align: left;
        }
        .excel-table tr:nth-child(even) td { background-color: #f9fbe7; }
    </style>
</head>
<body>

    <div style="width: 100%; max-width: 780px; display: flex; justify-content: flex-end; margin-bottom: 10px; gap: 8px;">
        <button class="btn btn-inactive" id="btn-open-admin" onclick="toggleAdminView()"> Админ-панель</button>
    </div>

    <div class="box-card" id="screen-register">
        <h2>Регистрация</h2>
        <p style="font-size: 14px;">Заполните данные перед прохождением анкеты</p>
        <div class="form-group">
            <label>Имя</label>
            <input type="text" id="reg-fname" class="input-field" placeholder="Введите имя">
        </div>
        <div class="form-group">
            <label>Фамилия</label>
            <input type="text" id="reg-lname" class="input-field" placeholder="Введите фамилию">
        </div>
        <div class="form-group">
            <label>Электронная почта</label>
            <input type="email" id="reg-email" class="input-field" placeholder="example@mail.ru">
        </div>
        <div class="error-text" id="reg-error"></div>
        <button class="btn btn-active" style="width: 100%;" onclick="submitRegister()">Перейти к опросу</button>
    </div>

    <div class="box-card" id="screen-survey" style="display: none;">
      <h2>Опрос будущего хозяина</h2>
        <p style="font-size: 13px; margin-bottom: 14px;">Ответьте на 11 вопросов — это поможет нам подобрать вам подходящего питомца.</p>
    
        <div class="form-group"><label>1. Вы проживаете в квартире или в частном доме?</label>
          <select id="q-housing-type" class="input-field">
            <option value="квартира">Квартира</option>
            <option value="частный дом">Частный дом</option>
          </select>
        </div>
    
        <div class="form-group"><label>2. Вы снимаете жилье, в котором живете?</label>
          <select id="q-housing-ownership" class="input-field">
            <option value="свое">Своё жильё</option>
            <option value="съемная">Съёмное жильё</option>
          </select>
        </div>
    
        <div class="form-group"><label>3. У вас есть разрешение от арендодателя?</label>
          <select id="q-landlord" class="input-field">
            <option value="да">Да</option>
            <option value="нет">Нет</option>
          </select>
        </div>
    
        <div class="form-group"><label>4. Ваш супруг/супруга не против вашего решения?</label>
          <select id="q-spouse" class="input-field">
            <option value="да">Да, не против</option>
            <option value="нет">Нет, есть возражения</option>
          </select>
        </div>
    
        <div class="form-group"><label>5. Есть ли у вас дети?</label>
          <select id="q-children" class="input-field">
            <option value="нет">Нет</option>
            <option value="да">Да</option>
          </select>
        </div>
    
        <div class="form-group"><label>6. У вас есть домашние животные?</label>
          <select id="q-pets" class="input-field">
            <option value="нет">Нет</option>
            <option value="да">Да</option>
          </select>
        </div>
    
      <div class="form-group"><label>7. У вас есть опыт с домашними животными?</label>
        <select id="q-experience" class="input-field">
            <option value="нет">Нет</option>
            <option value="да">Да</option>
        </select>
      </div>
    
      <div class="form-group"><label>8. Рассматриваете ли вариант стать опекуном животного?</label>
        <select id="q-guardianship" class="input-field">
            <option value="нет">Нет</option>
            <option value="да">Да</option>
        </select>
      </div>
    
      <div class="form-group"><label>9. Вы рассматриваете вариант забрать животное к себе?</label>
        <select id="q-adoption" class="input-field">
            <option value="нет">Нет</option>
            <option value="да">Да</option>
        </select>
      </div>
    
      <div class="form-group"><label>10. Если у вас был опыт с домашними животными, опишите какой?</label>
        <textarea id="q-experience-text" class="input-field" rows="3" placeholder="Например: 5 лет держали кота, 2 года ухаживали за собакой..."></textarea>
      </div>
    
      <div class="form-group"><label>11. Что сподвигло вас взять домашнее животное и зачем?</label>
        <textarea id="q-motivation-text" class="input-field" rows="3" placeholder="Расскажите о вашей мотивации..."></textarea>
      </div>
    
      <button class="btn btn-active" style="width: 100%;" onclick="submitSurvey()">Завершить опрос</button>
    </div>

    <div id="screen-tinder" style="display: none; width: 100%; flex-direction: column; align-items: center;">
        <div class="top-nav">
            <button class="btn btn-active" id="nav-cards" onclick="switchTinderTab('cards')"> Карточки</button>
            <button class="btn btn-inactive" id="nav-favs" onclick="switchTinderTab('favs')"> Избранное (0)</button>
            <button class="btn btn-inactive" onclick="logoutUser()">Выйти</button>
        </div>

        <div class="filter-nav" id="filter-bar">
            <button class="btn btn-active" id="flt-all" onclick="changeSpecies('all')">Все</button>
            <button class="btn btn-inactive" id="flt-cat" onclick="changeSpecies('cat')">Кошки</button>
            <button class="btn btn-inactive" id="flt-dog" onclick="changeSpecies('dog')">Собаки</button>
        </div>

        <div class="box-card" id="tinder-card-box">Загрузка карточек...</div>
        <div class="box-card" id="tinder-fav-box" style="display: none;"></div>
    </div>

    <div class="box-card admin-box" id="screen-admin" style="display: none;">
        <h2>Панель администратора</h2>
        <div id="admin-login-block">
            <p>Введите пароль администратора для доступа к таблице пользователей:</p>
            <div style="display: flex; gap: 10px; justify-content: center; max-width: 340px; margin: 0 auto;">
                <input type="password" id="admin-pass" class="input-field" placeholder="Пароль (admin2026)">
                <button class="btn btn-active" onclick="loginAdmin()">Войти</button>
            </div>
            <div class="error-text" id="admin-error" style="margin-top: 8px;"></div>
        </div>

        <div id="admin-data-block" style="display: none;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <button class="btn btn-active" onclick="triggerParserUpdate()"> Обновить карточки с сайта НИКА</button>
                <button class="btn btn-inactive" onclick="exportTableToCSV()"> Скачать для Excel (.csv)</button>
                <button class="btn btn-inactive" onclick="toggleAdminView()">Закрыть админку</button>
            </div>
            <div id="parser-status" style="font-size: 13px; margin-top: 8px; font-weight: bold;"></div>
            
            <div style="overflow-x: auto;">
                <table class="excel-table" id="users-excel-table">
                    <thead>
                        <tr>
                            <th>Почта</th>
                            <th>Имя</th>
                            <th>Фамилия</th>
                            <th>Вероятность стать обладателем питомца (%)</th>
                        </tr>
                    </thead>
                    <tbody id="admin-table-body"></tbody>
                </table>
            </div>
        </div>
    </div>
    <script>
        let currentUser = null;
        let currentSpecies = 'all';
        let petsList = [];
        let currentIndex = 0;
        let adminUsersCache = [];

        function updateScreenState() {
            const saved = localStorage.getItem('nika_current_user');
            const regEl = document.getElementById('screen-register');
            const srvEl = document.getElementById('screen-survey');
            const tndEl = document.getElementById('screen-tinder');

            if (!saved) {
                regEl.style.display = 'block';
                srvEl.style.display = 'none';
                tndEl.style.display = 'none';
                return;
            }

            currentUser = JSON.parse(saved);
            if (regEl) regEl.remove();

            if (!currentUser.survey_completed) {
                srvEl.style.display = 'block';
                tndEl.style.display = 'none';
            } else {
                if (srvEl) srvEl.remove();
                tndEl.style.display = 'flex';
                loadPetsFeed();
                refreshFavoritesList();
            }
        }

        async function submitRegister() {
            const first_name = document.getElementById('reg-fname').value;
            const last_name = document.getElementById('reg-lname').value;
            const email = document.getElementById('reg-email').value;
            const errEl = document.getElementById('reg-error');
            errEl.innerText = '';

            const res = await fetch('/api/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ first_name, last_name, email })
            });
            const data = await res.json();

            if (data.status === 'error') {
                errEl.innerText = data.message;
            } else {
                localStorage.setItem('nika_current_user', JSON.stringify(data.user));
                updateScreenState();
            }
        }

        async function submitSurvey() {
          const payload = {
            email: currentUser.email,
            housing_type: document.getElementById('q-housing-type').value,
            housing_ownership: document.getElementById('q-housing-ownership').value,
            landlord_permission: document.getElementById('q-landlord').value,
            spouse_consent: document.getElementById('q-spouse').value,
            has_children: document.getElementById('q-children').value,
            has_pets: document.getElementById('q-pets').value,
            has_experience: document.getElementById('q-experience').value,
            considers_guardianship: document.getElementById('q-guardianship').value,
            considers_adoption: document.getElementById('q-adoption').value,
            experience_text: document.getElementById('q-experience-text').value,
            motivation_text: document.getElementById('q-motivation-text').value
          };
          const res = await fetch('/api/survey', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
          });
          const data = await res.json();
          if (data.status === 'ok') {
            localStorage.setItem('nika_current_user', JSON.stringify(data.user));
            updateScreenState();
          }
        }

        function logoutUser() {
            localStorage.removeItem('nika_current_user');
            location.reload();
        }

        function getSkippedKey() { return `nika_skipped_${currentUser.email}`; }
        function getSkipped() { return JSON.parse(localStorage.getItem(getSkippedKey()) || '[]'); }
        function addSkipped(id) {
            const arr = getSkipped();
            if (!arr.includes(id)) {
                arr.push(id);
                localStorage.setItem(getSkippedKey(), JSON.stringify(arr));
            }
        }
        function resetSkipped() {
            localStorage.removeItem(getSkippedKey());
            loadPetsFeed();
        }

        function changeSpecies(species) {
            currentSpecies = species;
            ['all', 'cat', 'dog'].forEach(sp => {
                const btn = document.getElementById(`flt-${sp}`);
                btn.className = (sp === species) ? 'btn btn-active' : 'btn btn-inactive';
            });
            loadPetsFeed();
        }

        async function loadPetsFeed() {
            const box = document.getElementById('tinder-card-box');
            box.innerHTML = 'Загрузка карточек...';

            const [petsRes, favsRes] = await Promise.all([
                fetch(`/api/pets?species=${currentSpecies}`),
                fetch(`/api/favorites?email=${encodeURIComponent(currentUser.email)}`)
            ]);
            const allPets = await petsRes.json();
            const favs = await favsRes.json();

            const skipped = getSkipped();
            const favIds = new Set(favs.map(f => f.id));
            const seenImages = new Set(favs.map(f => f.image));

            petsList = [];
            for (const p of allPets) {
                if (!skipped.includes(p.id) && !favIds.has(p.id) && !seenImages.has(p.image)) {
                    seenImages.add(p.image);
                    petsList.push(p);
                }
            }
            currentIndex = 0;
            renderPetCard();
        }

        function renderPetCard() {
            const box = document.getElementById('tinder-card-box');
            if (currentIndex >= petsList.length) {
                box.innerHTML = `
                    <h3>Карточки в этой категории закончились</h3>
                    <p style="font-size: 14px;">Переключите фильтр сверху или сбросьте пропущенные анкеты.</p>
                    <button class="btn btn-inactive" onclick="resetSkipped()"> Показать пропущенных снова</button>
                `;
                return;
            }
            const p = petsList[currentIndex];
            box.innerHTML = `
                <div class="pet-card" id="active-pet-card">
                    <a href="${p.url}" target="_blank" title="Открыть полную страницу питомца">
                        <img src="${p.image}" alt="${p.name}">
                    </a>
                    <div style="font-size: 12px; margin-top: 6px;">Нажмите на фотографию для подробной информации о питомце</div>
                    <h2 style="margin: 10px 0 4px;">${p.name}</h2>
                    <p style="margin: 0; font-size: 14px;">${p.age}</p>
                </div>
                <div class="card-actions">
                    <button class="btn btn-inactive" onclick="skipPet()">Пропустить</button>
                    <button class="btn btn-active" onclick="likePet()">В избранное</button>
                </div>
            `;
        }
        function skipPet() {
            const p = petsList[currentIndex];
            document.getElementById('active-pet-card').classList.add('swipe-left');
            addSkipped(p.id);
            petsList = petsList.filter((item, i) => i <= currentIndex || (item.id !== p.id && item.image !== p.image));
            setTimeout(() => { currentIndex++; renderPetCard(); }, 350);
        }

        async function likePet() {
            const p = petsList[currentIndex];
            document.getElementById('active-pet-card').classList.add('swipe-right');
            addSkipped(p.id);
            petsList = petsList.filter((item, i) => i <= currentIndex || (item.id !== p.id && item.image !== p.image));

            await fetch('/api/favorite', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ...p, user_email: currentUser.email })
            });
            refreshFavoritesList();
            setTimeout(() => { currentIndex++; renderPetCard(); }, 350);
        }

        async function removeFavorite(id) {
            await fetch('/api/favorite/remove', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ id, user_email: currentUser.email })
            });
            refreshFavoritesList();
        }

        async function refreshFavoritesList() {
            const res = await fetch(`/api/favorites?email=${encodeURIComponent(currentUser.email)}`);
            const favs = await res.json();
            document.getElementById('nav-favs').innerText = ` Избранное (${favs.length})`;

            const box = document.getElementById('tinder-fav-box');
            if (!favs.length) {
                box.innerHTML = '<p>Список избранного пока пуст</p>';
                return;
            }
            box.innerHTML = favs.map(p => `
                <div class="fav-row">
                    <div style="display: flex; align-items: center;">
                        <img src="${p.image}">
                        <div>
                            <a href="${p.url}" target="_blank" style="font-weight: bold; color: #1b5e20; text-deсoration: none;">${p.name}</a>
                            <div style="font-size: 12px;">${p.age}</div>
                        </div>
                    </div>
                    <button class="btn btn-inactive" style="padding: 4px 10px;" onclick="removeFavorite('${p.id}')">✕</button>
                </div>
            `).join('');
        }

        function switchTinderTab(tab) {
            const isCards = tab === 'cards';
            document.getElementById('nav-cards').className = isCards ? 'btn btn-active' : 'btn btn-inactive';
            document.getElementById('nav-favs').className = !isCards ? 'btn btn-active' : 'btn btn-inactive';
            document.getElementById('filter-bar').style.display = isCards ? 'flex' : 'none';
            document.getElementById('tinder-card-box').style.display = isCards ? 'block' : 'none';
            document.getElementById('tinder-fav-box').style.display = !isCards ? 'block' : 'none';
            if (!isCards) refreshFavoritesList();
        }

        function toggleAdminView() {
            const adm = document.getElementById('screen-admin');
            const isHidden = adm.style.display === 'none';
            adm.style.display = isHidden ? 'block' : 'none';
            document.getElementById('btn-open-admin').className = isHidden ? 'btn btn-active' : 'btn btn-inactive';
        }

        async function loginAdmin() {
            const password = document.getElementById('admin-pass').value;
            const err = document.getElementById('admin-error');
            err.innerText = '';

            const res = await fetch('/api/admin/users', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ password })
            });

            if (!res.ok) {
                err.innerText = 'Неверный пароль администратора';
                return;
            }

            const data = await res.json();
            adminUsersCache = data.users;
            document.getElementById('admin-login-block').style.display = 'none';
            document.getElementById('admin-data-block').style.display = 'block';

            const tbody = document.getElementById('admin-table-body');
            tbody.innerHTML = adminUsersCache.map(u => `
                <tr>
                    <td>${u.email}</td>
                    <td>${u.first_name}</td>
                    <td>${u.last_name}</td>
                    <td><b>${u.adoption_probability !== null ? u.adoption_probability + '%' : 'Опрос не пройден'}</b></td>
                </tr>
            `).join('');
        }

        async function triggerParserUpdate() {
            const st = document.getElementById('parser-status');
            st.innerText = ' Идет обновление карточек с сайта НИКА...';
            const res = await fetch('/api/pets/refresh', { method: 'POST' });
            const data = await res.json();
            st.innerText = ` База обновлена! Загружено животных: ${data.count}`;
            if (currentUser && currentUser.survey_completed) loadPetsFeed();
        }

        function exportTableToCSV() {
            let csv = '\uFEFFПочта;Имя;Фамилия;Вероятность стать обладателем питомца (%)\n';
            adminUsersCache.forEach(u => {
                const prob = u.adoption_probability !== null ? `${u.adoption_probability}%` : 'Не пройден';
                csv += `${u.email};${u.first_name};${u.last_name};${prob}\n`;
            });
            const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
            const link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = 'users_nika_report.csv';
            link.click();
        }

        updateScreenState();
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
  return HTML_TEMPLATE


if __name__ == "__main__":
  webbrowser.open("http://127.0.0.1:8000")
  uvicorn.run(app, host="0.0.0.0", port=8000)