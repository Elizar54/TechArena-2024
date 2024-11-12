# Хакатон TechArena 2024 
## Команда «Три сигмы»

**Состав команды:**  
- [Елизарьев Ярослав Владимирович (ЭФ НГУ)](https://github.com/Elizar54)  
- [Метлин Александр Дмитриевич (ФИТ НГУ)](https://github.com/MetlinAlexander)  
- [Кузнецов Никита Сергеевич (ЭФ НГУ)](https://github.com/n-ikita)

### **Итог:** 3 место на хакатоне

![Фото команды](files/team_photo.jpg)
---

## Введение

Проект команды направлен на разработку алгоритма приближенного поиска ближайших соседей (ANN) для задач искусственного интеллекта. В современных алгоритмах точный поиск ближайших соседей может быть затратным по времени и памяти. Для ускорения мы выбрали приближенный метод поиска, жертвуя частично точностью ради улучшения производительности. Основной задачей было создать эффективный алгоритм, который помог бы оптимизировать задачи поиска, например, для LLM.

## Выбор алгоритма

Мы выбрали алгоритм IVFFlat и его модификацию IVFFlatPQ, которые являются одними из наиболее часто используемых методов ANN. IVFFlat отличается простотой, популярностью и эффективностью. На датасете Profiset-100k алгоритм IVFFlat демонстрирует хорошие метрики, особенно в сравнении с другими подходами.  

## Реализация алгоритма

### План реализации:

1. Реализация алгоритма KMeans
2. Обучение модели KMeans на датасетах
3. Сохранение размеченных данных
4. Сохранение обученной модели KMeans
5. Парсинг вектора для поиска соседей
6. Поиск 10 ближайших соседей

При тестировании на датасете SIFT мы установили оптимальное количество кластеров, минимизируя нагрузку на память. Для GIST мы использовали Flat Quantization для уменьшения памяти, занимая данные в формате `numpy.uint8`. Наш алгоритм позволял обрабатывать данные по батчам и выполнять поиск 10 ANN в пределах каждого кластера.

### Алгоритм поиска ближайших соседей:

1. Парсинг вектора для поиска
2. Предсказание кластера для вектора
3. Чтение датасета по батчам
4. Поиск соседей по евклидову расстоянию и вывод 10 ближайших индексов

## Результаты и выводы

Наш алгоритм показал хорошую метрику recall, но не достиг нужной скорости. Для улучшения возможно стоит рассмотреть уменьшение объема памяти или реализацию на языке C++.  
*Добавьте результаты тестов алгоритма*