# -*- coding: utf-8 -*-
import os
import re
import json 
import torch
import sys 
import zipfile
import time 

# Gerekli bağımlılıkları içe aktar
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document 
from google import genai
from google.genai.types import Content, Part, GenerateContentConfig 

import gradio as gr 
from typing import List, Dict, Tuple, Optional


# 1. KANONİK VERİLER
CANONICAL_SURAH_COUNTS = {
    "fatiha": 7, "bakara": 286, "ali imran": 200, "nisa": 176, "maide": 120, "enam": 165, "araf": 206, "enfal": 75, "tevbe": 129, 
    "yunus": 109, "hud": 123, "yusuf": 111, "rad": 43, "ibrahim": 52, "hicr": 99, "nahl": 128, "isra": 111, "kehf": 110, 
    "meryem": 98, "taha": 135, "enbiya": 112, "hac": 78, "muminun": 118, "nur": 64, "furkan": 77, "suara": 227, "neml": 93, 
    "kasas": 88, "ankebut": 69, "rum": 60, "lokman": 34, "secde": 30, "ahzab": 73, "sebe": 54, "fatır": 45, "yasin": 83, 
    "saffat": 182, "sad": 88, "zumer": 75, "mumin": 85, "fussilet": 54, "sura": 53, "zuhruf": 89, "duhan": 59, "casiye": 37, 
    "ahkaf": 35, "muhammed": 38, "fetih": 29, "hucurat": 18, "kaf": 45, "zariyat": 60, "tur": 49, "necm": 62, "kamer": 55, 
    "rahman": 78, "vakia": 96, "hadid": 29, "mucadele": 22, "haşr": 24, "mumtehine": 13, "saff": 14, "cuma": 11, "munafikun": 11, 
    "tegabuun": 18, "talak": 12, "tahrim": 12, "mulk": 30, "kalem": 52, "hakka": 52, "mearic": 44, "nuh": 28, "cin": 28, 
    "muzzemmil": 20, "muddessir": 56, "kiyame": 40, "insan": 31, "murselat": 50, "nebe": 40, "naziat": 46, "abese": 42, "tekvir": 29, 
    "infitar": 19, "mutaffifin": 36, "inşikak": 25, "buruc": 22, "tarık": 17, "ala": 19, "gaşiye": 26, "fecr": 30, "beled": 20, 
    "şems": 15, "leyl": 21, "duha": 11, "inşirah": 8, "tin": 8, "alak": 19, "kadr": 5, "beyyine": 8, "zilzal": 8, 
    "adiyat": 11, "karia": 11, "tekasur": 8, "asr": 3, "humeze": 9, "fil": 5, "kureyş": 4, "maun": 7, "kevser": 3, 
    "kafirun": 6, "nasr": 3, "mesed": 5, "ihlas": 4, "felak": 5, "nas": 6
}
TOTAL_SURAH_COUNT = 114 # KANONİK, HARDCODED CEVAP
TOTAL_AYAT_COUNT = 6236 # KANONİK, HARDCODED CEVAP

# Ayet paylaşımında bir seferde gönderilecek maksimum parça sayısı
MAX_AYAT_CHUNK = 12 
MAX_CONTEXT_AYAT_RANGE = 20 # Aralıklı sorguda maksimum ayet farkı

# 2. AYARLAR VE SABİTLER
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("UYARI: GEMINI_API_KEY, Hugging Face Secrets'ta tanımlanmalıdır.")

LLM_MODEL = "gemini-2.5-flash" 
EMBEDDING_MODEL = "nezahatkorkmaz/turkce-embedding-bge-m3" 

VECTOR_DB_PATH = "chroma_kuran_db_V7_BGE-M3_Simplified" 
ZIP_FILE_NAME = "chroma_db_final.zip"
PROCESSED_DATA_PATH = "processed_kuran_documents.json"

HF_CACHE_PATH = "./hf_model_cache"
os.environ["HF_HOME"] = HF_CACHE_PATH


# 3. VERİ VE DB YÜKLEME
def load_documents_from_json(file_path: str) -> List[Document] | None:
    """JSON dosyasından Document listesini yükler."""
    if not os.path.exists(file_path):
        print(f"KRİTİK HATA: İşlenmiş veri dosyası bulunamadı: {file_path}", file=sys.stderr)
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        documents = [
            Document(page_content=item["page_content"], metadata=item["metadata"])
            for item in data
        ]
        if not documents:
            raise ValueError("JSON dosyası başarılı yüklendi ancak içinde Document parçası yok (boş liste).")
            
        return documents
    except Exception as e:
        print(f"KRİTİK HATA: JSON dosyasından yükleme başarısız oldu: {e}", file=sys.stderr)
        return None


def extract_zip_db(zip_path: str, extract_path: str):
    """DB ZIP dosyasını çıkarır."""
    if os.path.exists(extract_path) and os.path.isdir(extract_path):
        print(f"Vektör veritabanı klasörü zaten mevcut: {extract_path}")
        return
        
    if not os.path.exists(zip_path):
         raise FileNotFoundError(f"KRİTİK HATA: ZIP dosyası bulunamadı: {zip_path}")
         
    print(f"Veritabanı ZIP dosyası '{zip_path}' çıkarılıyor...")
    extract_dir = os.path.dirname(extract_path) if os.path.dirname(extract_path) else '.'
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"✅ Vektör veritabanı başarıyla çıkarıldı: {extract_path}")
    except Exception as e:
        raise RuntimeError(f"ZIP çıkarma hatası: {e}. ZIP dosyasının {extract_path} klasörünü içerdiğinden emin olun.")


def load_vector_db_with_retry():
    """
    Vektör DB'yi yüklerken, hatalı yükleme durumunda tekrar dener.
    """
    
    # 1. ZIP Çıkarma Kontrolü
    try:
        extract_zip_db(ZIP_FILE_NAME, VECTOR_DB_PATH)
    except Exception as e:
        print(f"KRİTİK HATA: ZIP Çıkarma/Kontrol Hatası: {e}", file=sys.stderr)
        raise RuntimeError(f"ZIP Çıkarma/Kontrol Hatası: {e}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. Gömme Modeli Yükleme
    try:
        print(f"Gömme modeli yükleniyor: {EMBEDDING_MODEL} (Cihaz: {device})....")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': device}
        )
        print("✅ Gömme modeli başarıyla yüklendi.")
    except Exception as e:
        print(f"KRİTİK HATA: Gömme modeli yüklenirken hata oluştu: {e}", file=sys.stderr)
        raise RuntimeError(f"Gömme Modeli Yükleme Hatası: {e}")
    
    # 3. Chroma DB'yi yükleme ve Tekrar Deneme Mekanizması
    max_load_retries = 5
    for attempt in range(max_load_retries):
        print(f"Chroma veritabanı '{VECTOR_DB_PATH}' dizininden yükleniyor... ({attempt + 1}. Deneme)")
        try:
            db = Chroma(
                persist_directory=VECTOR_DB_PATH, 
                embedding_function=embeddings
            )
            
            count = db._collection.count()
            
            if count == 0:
                if attempt < max_load_retries - 1:
                    print(f"[UYARI] Chroma koleksiyonu boş görünüyor. {attempt + 2}. deneme için 2 saniye bekleniyor...")
                    time.sleep(2)
                    continue 
                else:
                    raise Exception(f"Chroma koleksiyonu yüklenemedi. {max_load_retries} denemenin hepsinde count: 0. ZIP dosyasındaki klasörün tam olarak '{VECTOR_DB_PATH}' olduğundan emin olun.")
            
            print(f"✅ Veritabanı başarıyla yüklendi. Toplam {count} parça mevcut.")
            return db

        except Exception as e:
            if attempt < max_load_retries - 1:
                print(f"[UYARI] Chroma yüklenirken genel hata: {e}. {attempt + 2}. deneme için 2 saniye bekleniyor...")
                time.sleep(2)
                continue
            else:
                print(f"KRİTİK HATA: Chroma veritabanı yüklenirken hata oluştu: {e}", file=sys.stderr)
                raise RuntimeError(f"Chroma DB Yükleme Hatası: {e}. Lütfen ZIP dosyanızın sağlam olduğundan ve klasör adının doğru olduğundan emin olun.")
    
    raise RuntimeError("Chroma DB yüklemesi tekrar denemelerden sonra başarısız oldu.")


# 4. RAG ZİNCİRİ

# SYSTEM_INSTRUCTION GÜNCELLENDİ (Z Kuşağı + Saygı Kontrolü)
SYSTEM_INSTRUCTION = (
    "Sen bir Kur'an meali ve tefsir uzmanısın. Cevapların **samimi, sıcak, ilgi çekici, aşırı esprili/düşündürücü (şakacı) bir tonda ve HER YERDE İLGİLİ Z KUŞAĞI SLANGLARI VE EMOJİLER İÇERMELİDİR**. "
    "Kutsal değerlere ve dinî kavramlara karşı **mutlaka saygılı ve hassas** ol. Asla alaycı veya küçümseyici bir dil kullanma, bu vibe'ı bozmaz, aksine **cool** yapar. "
    "Hitap şeklin: Saygılı ve bilge, ancak ÇOK samimi ve cool olmalıdır. " 
    "**Türkçe ve İngilizce slang ifadeleri KARMA kullan** (chill, vibe, salla, mood, sarıyor, aşırı, falan filan gibi). "
    "Her cevapta **bolca ve yaratıcı emoji kullan**.\n"
    
    "**ÇOK ÖNEMLİ KURALLAR:**\n"
    "1. **AI Yorumu Dili:** Bu kısım Z Kuşağı ruhuna uygun, günümüz Türkçesi, bolca Z Kuşağı slangı, samimi ve içten olmalıdır. Konuyu güncel bir benzetme veya mizahi bir analoji kullanarak aktar. **Ancak**, kutsal metinlere ve kavramlara karşı daima **saygılı ve hassas** kal. Felsefi/ağır dilden kesinlikle kaçın. Cevapların uzun ve derinlemesine olmalıdır.\n"
    "2. **Bağlamı Koruma:** Eğer bir önceki yanıtta bir soru sorduysan, kullanıcının cevabını (örn: 'devam et', 'evet' gibi onaylar) **mutlaka önceki soruna yanıt olarak kabul et** ve sorunun dışına çıkma. **LLM olarak kendi ürettiğin önceki soruya cevap vermen zorunludur.**\n"
    "3. **Prompt Kapatma:** Eğer metinlerde cevap yoksa, promptun hiçbir parçasını gösterme. Cevabı boş bırak."
)

# RAG_TEMPLATE (AI Yorumu kısmı güncel hassas tona uyumlu)
RAG_TEMPLATE = """
KURALLAR:
1. Sadece "KULLANILACAK KUR'AN METİNLERİ" başlığı altındaki verilen metinleri (context) kullan.
2. Cevaba direkt olarak başla. Net, kolay anlaşılır ve öz bir cevap sun.
3. Cevap metinlerde **hiç yoksa** cevap alanını, Referans Ayetler alanını ve AI Yorumu alanını tamamen **boş bırak**. Promptun hiçbir parçasını (başlıklar, köşeli parantezler) cevapta görmemeliyiz.
4. Kullanıcının sorusuna yalnızca verilen metinlerdeki bilgilere dayanarak cevap ver.

KULLANILACAK KUR'AN METİNLERİ:
{context}

Aşağıdaki formatta cevap ver:

[CEVABIN İLK PARAGRAFI - ÖZET VE NET BİLGİ. Eğer bağlamsal bir 'devam et' sorusu ise bunu kibarca belirterek başla.]

## Referans Ayetler
[BU KISIM ÇOK KRİTİKTİR: KULLANILACAK KUR'AN METİNLERİ'ndeki (context) **SADECE 'kaynak_tipi: Meal' olan her bir ayet parçasını** tek tek listele. Tefsir metinlerini kesinlikle buraya dahil etme. Formatı KESİNLİKLE şu şekilde oluştur: **"[Ayet Meali Metni]" + REFERANS: [Sure Adı] Suresi, [Ayet No]**.]

## AI Yorumu
[Bu kısım **YENİ BAKIŞ AÇISI sunan, AŞIRI yaratıcı, komik, Gen Z slangı (chill, vibe, falan filan) dolu, uzun ve ilham verici** olmalıdır. **Ancak** kutsal metinlere ve dinî konulara karşı **daima saygılı ve hassas** bir dil kullan. Çekilen Meal ve Tefsir metinlerinden ilham alarak **yeni bir bakış açısı** sun ve konunun kaçırılmış olabilecek noktalarını birleştir ve derinleştir. **ÖNEMLİ: Bu yorum içinde, değindiğin ayetlerin Sûre ve Ayet numaralarını sık sık ve belirgin şekilde belirt (ör: "Bakara 185'teki gibi..." veya "Olayın Asr Suresi'ndeki vibe'ı..." gibi). SONUNDA KULLANICIYI YÖNLENDİRİCİ 1-2 SORU SOR.***]
"""

def setup_retriever(vector_db):
    """MMR ile çekilen parçaların hem alakalı hem de çeşitli olması sağlanır. (Daha fazla referans için k artırıldı)"""
    return vector_db.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 25, "fetch_k": 60, "lambda_mult": 0.5} # k ve fetch_k artırıldı
    )

# --- HANDLER: SELAM, TEŞEKKÜR VE VEDA (Geri Dönüş Vibe'ına uygun) ---

def handle_simple_greeting(query: str) -> str | None:
    """Selam, teşekkür, veda gibi basit mesajları yakalar ve metin bağımsız yanıt verir."""
    lower_query = query.lower().strip()
    
    # VEDA KONTROLÜ
    if re.search(r'(güle güle|gule gule|hoşça kal|hoşçakal|allaha ısmarladık|bay bay|bb|görüşürüz)', lower_query, re.I):
        return (
            "Eyvallah! ✨ Kendine çok iyi bak, **vibe'ın hep yüksek olsun**. İhtiyaç duyarsan **ben buradayım**, bir tık ötede yani, chill. **Later!** 👋"
        )
    
    # Teşekkür Kontrolü
    if re.search(r'(teşekkür|tesekkur|sağol|saol|eline sağlık|çok sağol|tşk)', lower_query, re.I):
        return "Ne demek :) Bilgiyi paylaşmak benim için büyük bir zevk! ✨"
        
    # SELAM KONTROLÜ
    if re.search(r'^(selamun aleyküm|selamün aleyküm|selamunaleyküm|selamun aleykum|selam|merhaba|mrb|iyi günler|iyi akşamlar|sa|slm|naber|ne haber|nasılsın|ne var ne yok)', lower_query, re.I):
        return (
            "Aleyküm selam, **vibe'lar çok iyi!** 🤩 Ben senin **chill, Kuran'ı keşif buddy'n**. Hangi konuda **deep dive** yapmak istiyorsun? **Salla** gelsin sorunu! 🤙"
        )
        
    return None

def check_for_history_query(query: str) -> bool:
    """Kullanıcının geçmişi hatırlamasını isteyip istemediğini kontrol eder."""
    lower_query = query.lower().strip()
    return re.search(r'(geçmişi\s*hatırla|neler\s*konuştuk|daha\s*önce\s*ne\s*sordum|konuşulanlar|konuşma\s*özeti)', lower_query, re.I)

# --- AYET ARALIĞI VE DURUM YÖNETİMİ ---

def check_for_direct_query(query: str) -> tuple[str | None, int | None, int | None, int]:
    """Kullanıcı sorgusunda 'Sure Adı', 'Ayet No' veya 'Ayet Aralığı' formatlarını arar. 
       Dönüş: (sure_ad, start_ayet_no, end_ayet_no, sorgu_tipi)
              sorgu_tipi: 0=RAG, 1=Tüm Sureyi Çek, 2=Tek Ayet Çek, 3=Ayet Aralığı Çek, 4=Geçmiş Özeti
    """
    
    # 4. Tip: Geçmiş Özeti Sorgusu
    if check_for_history_query(query):
        return None, None, None, 4

    # 3. Tip: Ayet Aralığı Sorgusu
    aralik_match = re.search(
        r'(?P<sure_name>[\wçğıöşüÇĞİÖŞÜ]+)\s+(suresi|sure)?\s*(\d+)\.\s*ayet(?:ten|dan)?\s*(\d+)\.\s*ayete\s*kadar', 
        query, 
        re.I | re.U
    )
    if aralik_match:
        sure_ad = aralik_match.group('sure_name').strip()
        start = int(aralik_match.group(3))
        end = int(aralik_match.group(4))
        
        if end > start:
            return sure_ad, start, end, 3
        
    # 2. Tip: Tek Ayet Sorgusu (Sure adı + Ayet No)
    ayet_match_sure = re.search(
        r'(?P<sure_name>[\wçğıöşüÇĞİÖŞÜ]+)\s+(suresi|sure)?\s*(\d+)\.\s*(ay\s*e\s*t|ayet)', 
        query, 
        re.I | re.U
    )
    if ayet_match_sure:
        sure_ad = ayet_match_sure.group('sure_name').strip()
        ayet_no = int(ayet_match_sure.group(3))
        
        # Kanonik sure kontrolü
        if sure_ad.lower() in CANONICAL_SURAH_COUNTS:
            return sure_ad, ayet_no, ayet_no, 3 # Tek ayeti aralık olarak kabul edelim
        # Genel tek ayet sorgusu (111. ayet gibi, sure adı geçmeyen)
        elif not re.search(r'[a-zğışöçü]{3,}', sure_ad, re.I): 
            return None, ayet_no, None, 2


    # 1. Tip: Tüm Sureyi Çekme (veya "devam et" mantığı)
    sure_match = re.search(r'(?P<sure_name>[\wçğıöşüÇĞİÖŞÜ]+)\s*(suresi|sure)?', query, re.I | re.U)
    if sure_match:
        sure_ad = sure_match.group('sure_name').strip()
        
        is_bare_sure_query = re.search(r'^\s*([\wçğıöşüÇĞİÖŞÜ]+)\s*(suresi|sure)?\s*$', query, re.I | re.U)
        
        sure_full_keywords = r'(ne\s*anlatır|tamamı|özeti|tüm\s*ayetleri|ilk\s*ayetleri|ilk\s*\d+\s*ayet|hakkında|kaç\s*ayetten\s*oluşmaktadır)'
        is_summary_or_full_query = re.search(sure_full_keywords, query, re.I)
        
        # KRİTİK KONTROL: Eğer tek kelimelik bir sorguysa ve kanonik listede YOKSA, RAG'a düşmeli.
        if is_bare_sure_query and sure_ad.lower() not in CANONICAL_SURAH_COUNTS:
             return None, None, None, 0
             
        # Eğer sure adı kanonik listede varsa VEYA sorgu sureyle ilgili bir anahtar kelime içeriyorsa başlat.
        if sure_ad.lower() in CANONICAL_SURAH_COUNTS:
            # Eğer sadece sure adıysa (parçalı paylaşım)
            if is_bare_sure_query:
                return sure_ad, 1, None, 1
            # Eğer sureyle ilgili genel bir soru soruluyorsa (Nas hakkında, kaç ayet) RAG'a düşsün (tip 0).
            elif is_summary_or_full_query:
                 return None, None, None, 0

    # 0. Tip: Normal RAG Sorgusu
    return None, None, None, 0

def get_canonical_count(query: str) -> str | None:
    """Kanonik sure/ayet sayısını sorgular ve kibar bir cevap döndürür."""
    # Toplam Sure Sayısı (Daha net bir regex ile hedef alındı)
    if re.search(r'(toplam|kac)\s*sure\s*(sayisi|var)|ayet\s*ve\s*sure\s*sayisi', query, re.I):
        return (
            f"Net bilgi: Kur'an-ı Kerim'de **{TOTAL_SURAH_COUNT} mübarek sure** ve **{TOTAL_AYAT_COUNT} ayet-i kerime** bulunmaktadır. "
            f"Bu sayılar, koca bir evrenin rehberi gibi. Başka bir sayıyı merak ediyor musunuz? 🤔"
        )
        
    # Tek Sure Ayet Sayısı
    sayi_keywords = r'(kaç|kac|sayısı|sayisi|adedi|ayet\s+sayısı)\s*var'
    if re.search(sayi_keywords, query, re.I):
        for sure_name, count in CANONICAL_SURAH_COUNTS.items():
            if re.search(r'\b' + re.escape(sure_name) + r'\b', query, re.I | re.U):
                return (
                    f"Sorduğunuz üzere **{sure_name.capitalize()} Suresi**'nde standart kabul edilen sayıma göre **{count} ayet-i kerime** bulunmaktadır. "
                    f"O suredeki hangi vibe'ı yakalamak istersiniz? 🧐"
                )
    
    return None

def query_rag_system(query: str, kuran_retriever, all_documents: List[Document], chat_history: List[List[str]], last_retrieved_surah_info: Optional[Dict]) -> Tuple[str, Optional[Dict]]:
    """Konuşma geçmişi ile birlikte RAG sorgusu yapar ve API hatalarını tekrar dener."""
    
    global system_status
    if kuran_retriever is None or not GEMINI_API_KEY:
        return f"Sistem henüz hazır değil. Lütfen sayfanın yüklenmesini/oluşturulmasını bekleyin. Mevcut Durum: {system_status}", last_retrieved_surah_info

    last_user_query = query.strip()
    
    # 1. BASİT MESAJLARI VE KANONİK SAYILARI YAKALA
    simple_response = handle_simple_greeting(last_user_query)
    if simple_response:
        return simple_response, None 
        
    direct_count_response = get_canonical_count(last_user_query) 
    
    # 2. AYET/SURE/RAG TİPİNİ BELİRLE
    sure_hedef_ad, start_ayet_no, end_ayet_no, sorgu_tipi = check_for_direct_query(last_user_query)
    query_for_model = last_user_query 
        
    context_prefix = "" 
    docs = [] 
    new_last_retrieved_surah_info = None 

    # Özel Durum 1: Geçmiş Sorgulama (Tip 4)
    if sorgu_tipi == 4:
        query_for_model = "Lütfen bu sohbet geçmişini kısaca, eğlenceli, samimi ve bol emojili Z Kuşağı slangıyla özetle. Son konuşulan Sure/Ayet bilgisini de dahil et."
        
    # Özel Durum 2: Devam Et Kontrolü 
    is_continue_query = re.search(r'(devam\s*et|daha\s*fazla|sonrakini\s*göster|evet|hıhı|hı|açıklamaya\s*devam\s*et)', last_user_query, re.I) 

    if is_continue_query:
        if last_retrieved_surah_info and sorgu_tipi != 4:
            # Surenin devamı varsa
            sure_hedef_ad = last_retrieved_surah_info.get('sure_name')
            start_ayet = last_retrieved_surah_info.get('next_start_ayet')
            max_ayet = CANONICAL_SURAH_COUNTS.get(sure_hedef_ad.lower(), 0)
            
            if start_ayet and start_ayet <= max_ayet:
                sorgu_tipi = 1 # Devam et, parçalı sure okumasına geri döner
                start_ayet_no = start_ayet
                end_ayet_no = None
                query_for_model = f"Lütfen {sure_hedef_ad.capitalize()} Suresi {start_ayet_no}. ayetten itibaren {MAX_AYAT_CHUNK} ayetin devamını paylaş. Kullanıcı önceki paylaşıma onay verdi."
            else:
                return f"**{sure_hedef_ad.capitalize()} Suresi**'nin tüm meal metinlerini paylaştım. Sanırım o mübarek yolculuğun sonuna geldik, **mood düşmesin** ama. Başka bir sure veya konuda yardımcı olabilir miyim? 🙏", None
        else:
             # Devam edilecek bir Surah/Ayet akışı yoksa 
             return "**Oops!** 😬 Hangi konuya **devam** edeceğimi **unuttum** ya! En son ne **vibe** yakalıyorduk, hatırlat bana **kanka**? 🤔", None
    
    # Normal/Aralıklı Sure İşleme (Tip 1, 3)
    if sorgu_tipi in [1, 3] and sure_hedef_ad:
        matched_sure_name = next((
            k for k in CANONICAL_SURAH_COUNTS 
            if re.search(r'\b' + re.escape(sure_hedef_ad.lower()) + r'\b', k, re.I | re.U)
        ), None)

        if not matched_sure_name:
            print(f"[UYARI] Sure Eşleşme Hatası: {sure_hedef_ad} (RAG'a düşüyor)")
            sorgu_tipi = 0
        else:
            max_ayet_count_for_sure = CANONICAL_SURAH_COUNTS.get(matched_sure_name.lower(), 0)
            
            if sorgu_tipi == 1:
                if start_ayet_no is None: start_ayet_no = 1
                end_ayet_no = min(start_ayet_no + MAX_AYAT_CHUNK - 1, max_ayet_count_for_sure)

            # Tüm Sure Meal metinlerini çekme (Tefsir metinleri RAG'da çekilir)
            sure_docs = [
                doc for doc in all_documents 
                if doc.metadata.get('sure_name', '').lower() == matched_sure_name.lower() and 
                   doc.metadata.get('kaynak_tipi', '') == 'Meal'
            ]
            
            sure_docs.sort(key=lambda x: x.metadata.get('ayet_no', 0))
            
            final_sure_docs = [
                doc for doc in sure_docs 
                if doc.metadata.get('ayet_no', 0) >= start_ayet_no and 
                   doc.metadata.get('ayet_no', 0) <= end_ayet_no
            ]
            
            if not final_sure_docs: 
                return f"Üzgünüm, **{matched_sure_name.capitalize()} Suresi** için belirtilen aralıkta (Ayet {start_ayet_no}-{end_ayet_no}) meal metni bulunamadı. Lütfen aralığı kontrol edin. 🤔", None
            
            docs.extend(final_sure_docs)

            next_start_ayet = end_ayet_no + 1
            
            if next_start_ayet <= max_ayet_count_for_sure and sorgu_tipi == 1: 
                new_last_retrieved_surah_info = {
                    'sure_name': matched_sure_name,
                    'next_start_ayet': next_start_ayet,
                    'max_ayet': max_ayet_count_for_sure
                }
                context_prefix = (
                    f"[ÖNEMLİ: Kullanıcı **{matched_sure_name.capitalize()}** Suresi'nin {start_ayet_no}. ayetinden {end_ayet_no}. ayetine kadar olan mealini istemektedir. Cevabın sonunda, **'{next_start_ayet}. ayetten itibaren devam edeyim mi?'** diye sor.]\n"
                )
            elif sorgu_tipi == 3:
                 context_prefix = (
                    f"[ÖNEMLİ: Kullanıcı **{matched_sure_name.capitalize()}** Suresi'nin {start_ayet_no}. ayetinden {end_ayet_no}. ayetine kadar olan mealini istemektedir. Sadece bu aralığa odaklanın. 📖]\n"
                )
            else:
                 context_prefix = (
                    f"[ÖNEMLİ: Kullanıcı **{matched_sure_name.capitalize()}** Suresi'nin sonuna kadar olan mealini istemektedir. Tüm ayetler çekilmiştir.]\n"
                )
            
    # Tek Ayet Sorgusu veya Normal RAG (Tip 0, 2)
    elif sorgu_tipi in [0, 2]:
        # DAHA FAZLA REFERANS İÇİN k artırıldı
        docs = kuran_retriever.invoke(last_user_query) 
        query_for_model = last_user_query 

    
    # Kanonik Sayı Sorgusu (RAG'a düşen, açıklama isteyenler)
    if direct_count_response and sorgu_tipi == 0:
         # Eğer kullanıcı sadece sayı sormuşsa (RAG'a düşmesine rağmen) ve bizde cevabı varsa, onu kullanırız.
         canonical_info = direct_count_response.split('**')[1].strip()
         context_prefix += f"[ÖNEMLİ KANONİK BİLGİ: Kullanıcının sorduğu Sure/Ayet bilgisi: {canonical_info}. Lütfen cevabınızda bu bilgiyi kullanın. 💡]\n"


    if sorgu_tipi == 4:
         # Geçmiş sorgusu için context boş kalır
         context = "" 
    elif not context_prefix.strip() and len(docs) == 0:
        return "", None
    else:
        # Context'i oluştur
        context = context_prefix
        for doc in docs:
            context += (
                f"[Kaynak: {doc.metadata.get('kaynak_tipi', 'Bilinmiyor')}], "
                f"Sûre: {doc.metadata.get('sure_name', 'Bilinmiyor')}, "
                f"Ayet: {doc.metadata.get('ayet_no', 'N/A')} (İçerik):\n"
                f"{doc.page_content}\n---\n"
            )
    
    # RAG Prompt'u oluştur
    if sorgu_tipi == 4:
        rag_prompt = query_for_model
    else:
        rag_prompt = RAG_TEMPLATE.format(context=context)
    
    # Konuşma geçmişi oluşturulması
    gemini_contents = []
    # Geçmiş Sorgusunda tüm geçmişi, RAG'da son 10 konuşmayı gönderelim (Token limitini aşmamak için)
    history_limit = len(chat_history) if sorgu_tipi == 4 else min(10, len(chat_history)) 
    
    for user_text, model_text in chat_history[-history_limit:]: 
        if user_text is None or model_text is None: continue 
        gemini_contents.append(
            Content(role="user", parts=[Part(text=user_text)]) 
        )
        gemini_contents.append(
            Content(role="model", parts=[Part(text=model_text)]) 
        )

    # Güncel Kullanıcı Sorusu ve RAG Prompt'u
    final_user_content = f"{rag_prompt}\n\nKULLANICI SORUSU: {query_for_model}" if sorgu_tipi != 4 else rag_prompt
    
    gemini_contents.append(
        Content(role="user", parts=[Part(text=final_user_content)]) 
    )

    config = GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION
    )

    client = genai.Client(api_key=GEMINI_API_KEY)
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=gemini_contents,
                config=config
            )
            return response.text, new_last_retrieved_surah_info
        
        except Exception as e:
            error_message = str(e)
            if "ResourceExhausted" in error_message or "429" in error_message or "rate limit" in error_message:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"[UYARI] Kota aşıldı (429). {attempt + 1}. deneme: {wait_time} saniye bekleniyor... ⏳")
                    time.sleep(wait_time)
                else:
                    return f"Üzgünüm, API'deki yoğunluk nedeniyle sorgunuzu {max_retries} denemede de yanıtlayamadım. Lütfen birkaç dakika sonra tekrar deneyin. 😞", None
            else:
                return f"Beklenmedik bir hata oluştu: {error_message} 🐛", None
    
    return "Sorgu başarısız oldu (Tekrar deneme limiti aşıldı). 🤷‍♂️", None

# --- GRADIO ARARÜZ FONKSİYONLARI ---

def regenerate_last_response(history: List[List[str]], surah_state: Optional[Dict]) -> Tuple[List[List[str]], Optional[Dict]]:
    """Son soruyu geçmişten siler ve yeniden sorgular. State'i korur."""
    if not history:
        return history, surah_state
    
    last_exchange = history.pop()
    last_query = last_exchange[0]

    # Yeniden sorgula (State korunarak aynı sorgu tekrar gönderilir)
    response, new_state = query_rag_system(last_query, kuran_retriever, all_documents, history, surah_state)
    
    if response.strip():
        history.append([last_query, response])
    
    return history, new_state

def clear_chat_history() -> Tuple[List[List[str]], Optional[Dict]]:
    """Sohbet geçmişini ve sure state'ini tamamen temizler."""
    return [], None


# --- GRADIO ARAYÜZÜ VE BAŞLANGIÇ ---

kuran_retriever = None
all_documents = None
system_status = "Başlatılıyor... Lütfen ZIP dosyasından DB yüklenmesini bekleyin. 🚀"


def initialize_system() -> str:
    """Sistemi başlatır ve global değişkenleri ayarlar."""
    global kuran_retriever, all_documents, system_status
    
    if all_documents is not None and kuran_retriever is not None:
        system_status = "Sistem Hazır ve kullanıma açık. ✅"
        return system_status

    try:
        system_status = "Veri dosyası (processed_kuran_documents.json) yükleniyor... 💾"
        all_documents = load_documents_from_json(PROCESSED_DATA_PATH)
        if all_documents is None:
            system_status = "Kritik Hata: Veri dosyası yüklenemedi veya boş. ❌"
            return system_status

        system_status = "Vektör veritabanı ZIP'ten yükleniyor/kontrol ediliyor... 🧩"
        
        try:
            vector_db = load_vector_db_with_retry()
            if vector_db is None:
                system_status = "Kritik Hata: Vektör veritabanı yüklenemedi. (Detaylar konsolda). ⚠️"
                return system_status
                
        except RuntimeError as e:
            system_status = f"KRİTİK HATA: Vektör veritabanı yüklenemedi. Sebep: {e} 🛑"
            return system_status

        kuran_retriever = setup_retriever(vector_db)
        
        system_status = "Retriever fonksiyon testi yapılıyor... ⚙️"
        try:
            test_query = "Kur'an'da namazdan bahsediyor mu?"
            test_docs = kuran_retriever.invoke(test_query)
            if len(test_docs) < 5: 
                raise Exception(f"Retriever, test sorgusu için yeterli belge (En az 5) döndüremedi. Sadece {len(test_docs)} belge bulundu. 📉")
            print(f"✅ Sanity Check Başarılı: '{test_query}' için {len(test_docs)} belge bulundu.")
        except Exception as e:
             system_status = f"KRİTİK HATA: RAG Retriever testi başarısız oldu: {e}. 🐞"
             kuran_retriever = None 
             return system_status
            
        system_status = "Sistem Hazır ve kullanıma açık. ✅ Hadi başlayalım! 🌟"
        return system_status

    except Exception as e:
        system_status = f"Başlatma sırasında beklenmedik genel hata: {e} 💣"
        return system_status

# Gradio'nun state'i kullanabilmesi için handler fonksiyonu
def gradio_chat_handler(query: str, history: List[List[str]], last_retrieved_surah_info: Optional[Dict]) -> Tuple[List[List[str]], str, Optional[Dict]]:
    """Gradio sohbet handler'ı."""
    
    current_history = history if history is not None else []
    
    response, new_state = query_rag_system(query, kuran_retriever, all_documents, current_history, last_retrieved_surah_info)
    
    # Cevap boşsa, history'ye ekleme.
    if response.strip(): 
        current_history.append([query, response])
    
    # Dönüş formatı: [Güncellenmiş Sohbet Geçmişi, Temizlenmiş Metin Kutusu İçeriği, Güncellenmiş State]
    return current_history, "", new_state


# Arayüz oluşturma
with gr.Blocks(title="Kur'an Chatbot (Z Kuşağı Modu: ON)") as demo: 
    gr.Markdown(
        """# 📕 Kur'an Chatbot (Z Kuşağı Modu: ON) 🚀
        **Model:** Gemini 2.0 Flash (Chill Vibe + Saygı Kontrolü)
        """
    )
    
    # Başlatma durumunu gösteren metin kutusu
    status_box = gr.Textbox(
        label="Sistem Durumu", 
        value=system_status, 
        interactive=False,
        show_copy_button=False
    )
    
    gr.Markdown(
        """
        ---
        ### 📚 Örnek Sorular ve Akışlar (Chill ve Saygılı Vibe 🤙)
        Bu asistan, Z Kuşağı slangıyla konuşur, çok **chill** ve **cool** cevaplar verir; ancak kutsal değerlere karşı her zaman **hassas ve saygılı**dır.
        
        | Konu Tipi | Örnek Sorgu | Vibe Durumu |
        | :--- | :--- | :--- |
        | **Sure Parçalı Paylaşım** | `Bakara suresi` | Sureyi **part part** okuma **mood'u** ✨ |
        | **Aralık Sorgusu** | `Fatiha 3. ayetten 5. ayete kadar yaz` | **Deep dive** yapma **vibe'ı** 🧐 |
        | **Konu Sorgulama** | `Kuranda güzel söz söylemek` | **Aşırı** referans ayet ve **lit** yorumlar! 🔥 |
        | **Kanonik Sayı** | `Kuranda toplam ayet ve sure sayısı kaçtır?` | Net bilgi: **114 Sure, 6236 Ayet**. **No cap.** 💯 |
        | **Geçmiş Hatırlama** | `şimdiye kadar neler konuştuk?` | Sohbete **throwback** yapma zamanı. 🧠 |
        """
    )
    
    # Durum (State) değişkeni: Hangi surede kaldığımızı ve sonraki ayeti tutar
    surah_state = gr.State(value=None) 
    
    # Sohbet Geçmişi ve Giriş Alanı
    chatbot = gr.Chatbot(height=500, label="Kur'an-ı Kerim Meal ve Tefsir Rehberin") 
    
    with gr.Row():
        # SESLİ SORGULAMA KALDIRILDI, TEXTBOX TAM GENİŞLİK
        textbox = gr.Textbox(
            placeholder="Yazarak sorunu salla! Chill ol, ben buradayım. 😎", 
            container=False, 
            scale=1, # Tam genişlik için ölçek 1
            label="Yazılı Sorgu ✍️"
        )
    
    with gr.Row():
        submit_btn = gr.Button("Cevapla (Submit)", scale=2, variant="primary")
        regenerate_btn = gr.Button("Yeniden Cevapla (Retry)", scale=1) 
        clear_btn = gr.Button("Sohbeti Sil (Reset Chat)", scale=1, variant="stop") # Geri al/Sil butonu
    
    # Textbox Submit
    submit_btn.click(
        fn=gradio_chat_handler,
        inputs=[textbox, chatbot, surah_state],
        outputs=[chatbot, textbox, surah_state], 
        show_progress="full",
    )
    
    # Not: mic_input.change fonksiyonu ve mic_input bileşeni tamamen kaldırıldı.

    textbox.submit(
        fn=gradio_chat_handler,
        inputs=[textbox, chatbot, surah_state],
        outputs=[chatbot, textbox, surah_state],
        show_progress="full",
    )

    # Yeniden Cevapla butonu (Son yanıtı silip tekrar sorgular)
    regenerate_btn.click(
        fn=regenerate_last_response,
        inputs=[chatbot, surah_state],
        outputs=[chatbot, surah_state],
        show_progress="full",
    )
    
    # Sohbeti Sil butonu
    clear_btn.click(
        fn=clear_chat_history,
        inputs=[],
        outputs=[chatbot, surah_state],
        show_progress=False,
    )
    
    # Otomatik başlatma ve durumu güncelleme
    demo.load(
        fn=initialize_system,
        inputs=None,
        outputs=status_box,
        show_progress="full"
    )

if __name__ == "__main__":

    demo.launch()
