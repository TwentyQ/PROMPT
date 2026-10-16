'''Блок №1 - объяснение термина'''
from gigachat import GigaChat

with GigaChat(credentials="OThhZGViNTgtN2E0Mi00YmExLTgzMTctM2YwNjFmNGI0NzNkOmM2YzYzMGJlLTczMGQtNDk3MC04MjRlLWQwZjBkZWRkM2U5Mg==",
              ca_bundle_file=r"C:\Users\Anni\Downloads\windows_russian_trusted_root_ca/russian_trusted_root_ca.cer",model='Gigachat-2-Pro', scope='GIGACHAT_API_B2B') as giga:
   response_professional_activity = giga.chat("Что такое посадка?")
   print(response_professional_activity.choices[0].message.content)
   print('-' * 134)
   response_household_activities = giga.chat("Что такое парогенератор у кофемашины?")
   print(response_household_activities.choices[0].message.content)
   print('-' * 134)
   response_educational_activities = giga.chat("Что такое кинетическая энергия?")
   print(response_educational_activities.choices[0].message.content)


'''Блок №2 - резюмирование текста'''
from gigachat import GigaChat

with GigaChat(credentials="OThhZGViNTgtN2E0Mi00YmExLTgzMTctM2YwNjFmNGI0NzNkOmM2YzYzMGJlLTczMGQtNDk3MC04MjRlLWQwZjBkZWRkM2U5Mg==",
              ca_bundle_file=r"C:\Users\Anni\Downloads\windows_russian_trusted_root_ca/russian_trusted_root_ca.cer",model='Gigachat-2-Pro', scope='GIGACHAT_API_B2B') as giga:
   file_professional_activity = open('news_text.txt', encoding='utf-8')
   text_professional_activity = file_professional_activity.read()
   response_professional_activity = giga.chat(f'{text_professional_activity} "Прорезюмируй этот текс"')
   print(response_professional_activity.choices[0].message.content)
   print('-' * 134)
   file_household_activities = open('news_text.txt', encoding='utf-8')
   text_household_activities = file_household_activities.read()
   response_household_activities = giga.chat(f'{text_household_activities} "Прорезюмируй этот текс"')
   print(response_household_activities.choices[0].message.content)
   print('-' * 134)

'''Блое №3 - вопросы пользователя по тексту'''
from gigachat import GigaChat

with GigaChat(credentials="OThhZGViNTgtN2E0Mi00YmExLTgzMTctM2YwNjFmNGI0NzNkOmM2YzYzMGJlLTczMGQtNDk3MC04MjRlLWQwZjBkZWRkM2U5Mg==",
              ca_bundle_file=r"C:\Users\Anni\Downloads\windows_russian_trusted_root_ca/russian_trusted_root_ca.cer",model='Gigachat-2-Pro', scope='GIGACHAT_API_B2B') as giga:
   file_professional_activity = open('rbk_text.txt', encoding='utf-8')
   text_professional_activity = file_professional_activity.read()
   response_professional_activity = giga.chat(f'{text_professional_activity} "Насколько снизился официальный курс доллара с начала года? ответь по тексту"')
   print(response_professional_activity.choices[0].message.content)
   print('-' * 134)
   file_household_activities = open('rose_text.txt', encoding='utf-8')
   text_household_activities = file_household_activities.read()
   response_household_activities = giga.chat(f'{text_household_activities} "Какие розы цветут один раз? ответь по тексту"')
   print(response_household_activities.choices[0].message.content)
   print('-' * 134)
   file_educational_activities = open('study_text.txt', encoding='utf-8')
   text_educational_activities = file_educational_activities.read()
   response_educational_activities = giga.chat(f'{text_educational_activities} "Когда Невский одержал победу? ответь по тексту"')
   print(response_educational_activities.choices[0].message.content)

