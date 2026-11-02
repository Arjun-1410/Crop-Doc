CropDoc AI: AI crop doctor for Hyderabad farmers. 10-slide project and startup deck. Audience: university faculty, incubator panel. Keep each slide short, use icons and simple visuals, green agriculture theme.

---
# CropDoc AI
Snap a leaf photo. Get the disease and the cure in Telugu, in seconds.
Team: Arjun Dosapati (24WU0102184), Sagar (4036), Vishwapal (2157), Kartheek (2154), Woxsen University
Live app built and tested, November 2026

---
# The problem and our solution
Problem:
- Farmers spot crop disease late, after it spreads
- Agriculture officers are far away; advice takes days
- Wrong pesticide wastes money and harms soil
Solution:
- Phone photo of the leaf, AI names the problem in under 1 second
- Advice in Telugu text and voice, with exact spray dose
- "Not sure? Ask an expert" safety gate

---
# What we built: a working app
- Pick crop: paddy, cotton, maize, chilli, tomato
- Real AI diagnosis of 36 diseases, pests and nutrient problems
- Confidence bands: Confident / Possible / Not sure
- Wrong-crop and blurry-photo warnings
- Spray calculator: per litre, per 16 L tank, per field
- Telugu voice, WhatsApp "ask expert" message, history, farmer feedback
- AI runs on the phone: photos never leave the device

---
# How the AI works
1. Data: 18,359 leaf photos, Indian field datasets (Paddy Doctor, SAR-CLD-2024 cotton, chilli, maize, tomato)
2. Model: MobileNetV3, pretrained on ImageNet, fine-tuned for 4 epochs
3. Honest test on 1,833 photos the AI never saw
4. Exported to ONNX, runs in the phone browser in 60 to 180 ms
5. Farmer feedback becomes new training data

---
# Results
- Overall accuracy: 93.6% on unseen photos
- When the AI is confident: 97.7% correct (88% of photos)
- Tomato 98.4% · Maize 95.9% · Cotton 95.3% · Paddy 87.5% · Chilli 80.0%
- Speed: under 0.2 s per photo (target was 2 s)
- Next: test on real Telangana farm photos

---
# Built for Hyderabad and Telangana
- Spray doses from PJTAU and ICAR advisories (e.g. paddy blast: tricyclazole 0.6 g/L)
- Banned antibiotics (streptocycline) removed from advice
- Season banner: Vanakalam (Jun–Oct), Yasangi (Nov–Mar)
- Help: Kisan Call Centre 1800-180-1551 (Telugu), KVK Ranga Reddy (CRIDA Hayathnagar), village AEO at Rythu Vedika

---
# Business model to scale as a startup
Free for farmers, paid by partners (learning from Plantix, which is free and earns via retailers)
- Farmers: free 5 scans per week; Pro ₹99/month for unlimited scans and expert chat
- FPOs: ₹5,000/month for member dashboards and bulk advisory
- Agri-input dealers: partner app with verified prescriptions and stock leads
- Government and CSR: disease-surveillance dashboards for the Agriculture Department and KVKs (Saagu Baagu model)
- Target: about 80% gross margin; costs are cloud, expert labelling, field staff

---
# Go-to-market and scale plan
- Month 1–3: pilot with 30 farmers and 1 FPO near Siddipet / Ranga Reddy; expert-review all advice
- Month 3–6: Android app with offline mode; 2,000+ local field photos; 1,000 users
- Month 6–12: 5,000 farmers, 5 crops, first FPO and dealer contracts
- Year 2: all Telangana districts via Saagu Baagu and FPO networks; add red gram, groundnut
- Year 3: Andhra Pradesh and Karnataka (Telugu and Kannada)
Edge: Telangana-verified advice, on-phone AI, Telugu voice, dose calculator

---
# Who we are reaching out to
Expert validation and field trials:
- KVK Ranga Reddy (ICAR-CRIDA, Hayathnagar)
- PJTAU, Rajendranagar (AgHub innovation hub)
Incubation and funding:
- a-IDEA, ICAR-NAARM: KrishiBoot incubation, RKVY-RAFTAAR grants (₹5 L to ₹25 L), SIDBI seed fund
- ABI-ICRISAT, Patancheru: NIDHI seed support (up to ₹50 L)
- T-Hub, Hyderabad
Scale partners:
- Telangana Agriculture Department and Digital Green (Saagu Baagu)
- FPOs and agri-input dealers around Siddipet and Ranga Reddy

---
# Next steps and our ask
This month:
- Expert review of all 36 advice entries with KVK / PJTAU
- Field test with 10 to 20 farmers near Hyderabad
- Apply to a-IDEA KrishiBoot and ABI-ICRISAT
Our ask: ₹8 lakh seed funding: 35% field data and labelling, 30% cloud, 25% app, 10% outreach
Thank you · Questions?
