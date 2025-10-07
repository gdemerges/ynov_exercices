# 1. Créer une liste de 1 à 10
liste = list(range(1, 11))
print("1. Liste :", liste)

# 2. Ajouter un élément à la fin (11)
liste.append(11)
print("2. Ajout :", liste)

# 3. Insérer un élément au début (0)
liste.insert(0, 0)
print("3. Insertion :", liste)

# 4. Supprimer le troisième élément
liste.pop(2)
print("4. Suppression du 3ème élément :", liste)

# 5. Inverser la liste
liste.reverse()
print("5. Inversion :", liste)

# 6. Trier la liste
liste.sort()
print("6. Tri :", liste)

print("------------------------------------------------------------------------")

# 1. Créer un dictionnaire d’étudiants
etudiants = {"Alice": 18, "Bob": 20, "Charlie": 17, "Diana": 19}
print("Dictionnaire :", etudiants)

# 2. Ajouter un étudiant
etudiants["Eve"] = 21
print("Ajout : ", etudiants)

# 3. Supprimer Charlie
del etudiants["Charlie"]
print("Suppression :", etudiants)

# 4. Créer un nouveau dictionnaire avec les étudiants majeurs
etudiants_majeurs = {nom: age for nom, age in etudiants.items() if age >= 18}
print("Étudiants majeurs :", etudiants_majeurs)
