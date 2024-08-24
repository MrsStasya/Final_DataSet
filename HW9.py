# Задание 1.
# Выберите датасет, который имеет отношение к вашей области интересов или исследований.
#  Датасет должен содержать неструктурированные данные, требующие разметки для решения конкретной задачи,
#  например, анализа настроений или распознавания именованных сущностей.

# Задание 2.
# Выполните разметку на основе правил (rule-based labeling) на подмножестве выбранного датасета.
#  Разработайте и реализуйте набор правил или условий, которые позволят автоматически присваивать метки данным на основе 
# определенных шаблонов или критериев.

# Задача 3.
# Выполните разметку вручную отдельного подмножества выбранного датасета с помощью выбранного вами инструмента разметки.

# Задача 4.
# Объедините данные, размеченные вручную, с данными, размеченными на основе правил. Объедините два
#  подмножества размеченных данных в один набор данных, сохранив при этом соответствующую структуру и целостность.

# Задача 5.
# Обучите модель машинного обучения, используя объединенный набор размеченных данных. 
# Разделите датасет на обучающий и тестовый наборы и используйте обучающий набор для обучения модели.

# Задача 6.
# Оценить эффективность обученной модели на тестовом датасете. Используйте подходящие метрики оценки. 
# Интерпретируйте результаты и проанализируйте эффективность модели в решении задачи разметки.

import pandas as pd
from sklearn.model_selection import train_test_split
#машинное обучение
from sklearn.model_selection import train_test_split
# Пример модели
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Выполнение задания 1
# Загружаем дата сет
data = pd.read_csv("HealthData.csv")

# Выводим 5 первых столбцов
print(data.head())

# Выполнение задания 2
# Будем работать со столбцом Body Temp. Нормальная температура по F = 98 градусов
def rule_based_labeling(row):
    if row["BodyTemp"] > 98:
        return "High"
    else:
        return "Normal"

# Добавляем новый столбец в дата сет с рейтингом
data["Rating_BodyTemp"] = data.apply(rule_based_labeling, axis=1)
print(data.head())

# # Разделим файл на 2 подмножества
set_auto, set_manual = train_test_split(data, test_size=0.05, random_state=42)

# Сохраним второе подмножество
set_manual.to_csv('set_manual.csv', index=False)
print(set_auto)

# Присвоение начальных меток
set_auto['RiskLevel'] = 'normal'

# Условия для высокого риска
set_auto.loc[(set_auto['Age'] < 5) & ((set_auto['SystolicBP'] > 95) | (set_auto['DiastolicBP'] > 65) | (set_auto['BodyTemp'] > 98)), 'RiskLevel'] = 'high'
set_auto.loc[(set_auto['Age'] < 20) & ((set_auto['SystolicBP'] > 105) | (set_auto['DiastolicBP'] > 70) | (set_auto['BodyTemp'] > 98)), 'RiskLevel'] = 'high'
set_auto.loc[(set_auto['Age'] >= 20) & (set_auto['Age'] < 40) & ((set_auto['SystolicBP'] > 120) | (set_auto['DiastolicBP'] > 80) | (set_auto['BodyTemp'] > 98)), 'RiskLevel'] = 'high'
set_auto.loc[(set_auto['Age'] >= 40) & (set_auto['Age'] < 90) & ((set_auto['SystolicBP'] > 125) | (set_auto['DiastolicBP'] > 84) | (set_auto['BodyTemp'] > 98)), 'RiskLevel'] = 'high'

# Условия для среднего риска
set_auto.loc[(set_auto['Age'] >= 35) & (set_auto['Age'] <= 50) & ((set_auto['SystolicBP'] > 120) | (set_auto['DiastolicBP'] > 80) | (set_auto['BS'] > 120) | (set_auto['BodyTemp'] > 37.5) | (set_auto['HeartRate'] > 90)), 'RiskLevel'] = 'low'
print(set_auto)

# Сохраним 
set_auto.to_csv('set_auto.csv', index=False)

# Выполнение задания 3
# Загружаем дата сет
data_manual = pd.read_csv("set_manual.csv")

# Выводим 5 первых столбцов
print(data_manual.head())

# Далее производим манипуляции с помощью Label Studio. У меня она установлена. Запускаем в терминале команду label-studio, которая открывает 
# окно интернет браузера
# далее проходим регистрацию/входим в акаунт. Загружаем файл "set_manual.csv" вбиваем теги и вручную на каждой строке проставляем. Сохраняем, 
#выгружаем в папку. У меня назван как set_manualUpdate.csv (до конца файла не проставляла)
# Прочитаем файл и распечатаем первые 5 строк для проверки
data_manual = pd.read_csv("set_manualUpdate.csv")
print(data_manual.head())

# Выполнение задания 4
set1 = pd.read_csv('set_auto.csv')
set2 = pd.read_csv('set_manualUpdate.csv')

# Объедините два дата сета в один 
merge_set = pd.concat([set1, set2], ignore_index=True)

#Удалим ненужные нам столбцы
merge_set.drop('annotation_id', axis=1, inplace=True)
merge_set.drop('annotator', axis=1, inplace=True)
merge_set.drop('created_at', axis=1, inplace=True)
merge_set.drop('id', axis=1, inplace=True)
merge_set.drop('lead_time', axis=1, inplace=True)
merge_set.drop('sentiment', axis=1, inplace=True)
merge_set.drop('updated_at', axis=1, inplace=True)

# Почистим данные в столбце RiskLevel (нужно заменить low risk на low и т.д. - мне же лень было полностью руками в Label Studio идти по файлу)
merge_set['RiskLevel'] = merge_set['RiskLevel'].replace(to_replace='high risk', value= 'high')
merge_set['RiskLevel'] = merge_set['RiskLevel'].replace(to_replace='low risk',value= 'low')
merge_set['RiskLevel'] = merge_set['RiskLevel'].replace(to_replace='mid risk', value='normal')

# Сохраним в csv
merge_set.to_csv('merge_set.csv', index=False)
merge_set = pd.read_csv('merge_set.csv')

# Выполнение задания 5
# Замена строк на числа в датасете
pd.set_option('future.no_silent_downcasting', True)
merge_set['Rating_BodyTemp'] = merge_set['Rating_BodyTemp'].replace({'Normal':0, 'High':1})
merge_set['RiskLevel'] = merge_set['RiskLevel'].replace({'low':0, 'high':1, 'normal':2})


merge_set.to_csv('merge_set.csv', index=False)
merge_set = pd.read_csv('merge_set.csv')
print(merge_set.head())

# Разделите данные на признаки (X) и целевую переменную (y)
x = merge_set.drop('BodyTemp', axis=1) 
y = merge_set['Rating_BodyTemp']

# Разделим данные на обучающий и тестовый наборы. test_size и random_state указываем любые
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)  

# Инициализируем модель машинного обучения (в данном случае пример с логистической регрессией)
model = LogisticRegression() 

# Обучим модель на обучающем наборе
model.fit(x_train, y_train)

# Оценим производительность модели на тестовом наборе
accuracy = model.score(x_test, y_test)
print(accuracy)

# Выполнение задания 6

# Предсказание меток классов на тестовом наборе
y_pred = model.predict(x_test)

# Оценка точности модели
accuracy_1 = accuracy_score(y_test, y_pred)
print(accuracy_1)

# Classification Report для полного анализа метрик 
print("Classification Report для полного анализа метрик ")
print(classification_report(y_test, y_pred))

# Матрица ошибок (Confusion Matrix) для визуализации количества правильных и неправильных прогнозов
print("Confusion Matrix")
confusion_matrix(y_test, y_pred)