import os
import numpy as np

froms = []
cont = []
def process_text_files(folder_path):
    # Получаем список всех файлов в папке
    for filename in os.listdir(folder_path):
        contacts = []
        # Проверяем, является ли файл текстовым файлом
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)

            # Открываем и читаем содержимое файла
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('from'):
                        with open('messages/contacts.txt','r') as g:
                            for line1 in g:
                                line1 = [line1.strip()]
                                contacts += line1

                        parts = line.strip().split(':')
                        froms.append(parts[1]) if parts[1] not in contacts else cont.append(parts[1])
        check_email(filename)

def act(x):
    return 1 if x >= 0.5 else 0


def check_email(filename):
    key = 0
    im = 0
    num_of_messages = 0
    num_of_mes_cont = 0

    k_words = 'распродажа,покупайте,скидка,магазин,супер-распродажа,мега-распродажа,повезло,успейте,ограничена,персональная,предложение,выгодное,выгодные,выгодный,выгода'.split(
        ',')
    with open(folder_path + '\\'+filename, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith('from'):
                num_of_messages = froms.count(line.strip().split(':')[1])
                num_of_mes_cont = cont.count(line.strip().split(':')[1])
            if line.startswith('message'):
                mes = line.strip().split(':')[1]
                for i in mes.strip().split(' '):
                    if i in k_words:
                        key += 1
                    elif i == '*image*':
                        im += 1
    if key + im + num_of_messages + num_of_mes_cont != 0:
        x = np.array([key/(key+im+num_of_messages), im/(key+im+num_of_messages),
                      num_of_messages/(key+im+num_of_messages), num_of_mes_cont/(key+im+num_of_messages)])
        w11 = [0.52, 1, 0.7, -1]
        w12 = [0.2, 0.1, 0.1, 1]
        weight1 = np.array([w11, w12])
        weight2 = np.array([1, 0])
        sum_hidden = np.dot(weight1, x)
        out_hidden = np.array([act(x) for x in sum_hidden])
        sum_end = np.dot(weight2, out_hidden)
        y = act(sum_end)
        print(f'{filename} is spam') if y == 1 else print(f'{filename} isn\'t spam')
    else:
        print('not spam')



# Пример использования функции
folder_path = r'C:\Users\Admin\PycharmProjects\AIlearning\AI_classes\Introdusing\spam_detector\messages'  # Замените на путь к вашей папке
process_text_files(folder_path)

