# Пример: явный DI и читаемый пайплайн

Читай при создании нового сценария или выделении этапов из крупного обработчика.
Пример иллюстрирует границы компонентов; он не требует переносить эти имена и число классов
в каждую задачу. Код совместим с Python 3.12 и использует только стандартную библиотеку.

## Как увидеть пайплайн в любой задаче

Сначала назови исходные данные и ожидаемый результат, затем вырази путь между ними
смысловыми шагами. Верхняя функция или метод становятся читаемым описанием этого пути.

| Задача | Сценарий на верхнем уровне |
| --- | --- |
| Регистрация | проверить запрос → подготовить учётную запись → сохранить → сформировать ответ |
| Расчёт стоимости | подготовить позиции → применить правила цен → рассчитать итог |
| Формирование отчёта | загрузить показатели → агрегировать → построить представление |
| CLI-команда | разобрать аргументы → выполнить операцию → вывести результат |

Шаг с собственной сложностью раскрывай аналогичным образом уровнем ниже. Ветвления
и циклы оставляй видимыми в управляющем методе. Число шагов определяется задачей;
не подгоняй все сценарии под один шаблон «прочитать → сохранить».

## Сценарий и его контракты

```python
import dataclasses
import typing


@typing.final
@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class ParsedDocument:
    body_text: str


class DocumentReader(typing.Protocol):
    async def fetch_document(self, document_key: str) -> bytes: ...


class DocumentWriter(typing.Protocol):
    async def save_document(
        self,
        document_key: str,
        parsed_document: ParsedDocument,
    ) -> None: ...


@typing.final
@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class DocumentPipeline:
    document_reader: DocumentReader
    document_writer: DocumentWriter

    async def process_document(self, document_key: str) -> ParsedDocument:
        document_bytes: typing.Final = await self.document_reader.fetch_document(document_key)
        parsed_document: typing.Final = parse_document(document_bytes)
        await self.document_writer.save_document(document_key, parsed_document)
        return parsed_document


def parse_document(document_bytes: bytes) -> ParsedDocument:
    body_text: typing.Final = document_bytes.decode("utf-8").strip()
    if not body_text:
        raise ValueError("Document must contain text")
    return ParsedDocument(body_text=body_text)


def build_document_pipeline(
    *,
    document_reader: DocumentReader,
    document_writer: DocumentWriter,
) -> DocumentPipeline:
    return DocumentPipeline(
        document_reader=document_reader,
        document_writer=document_writer,
    )
```

Здесь верхний метод показывает весь порядок обработки. Чистый parser остаётся функцией;
он не нуждается в классе или Protocol. Reader и writer — заменяемые границы I/O. Их
конкретные адаптеры и соединения создаются в точке запуска приложения, где ресурсам
назначается владелец. Для маленького проекта отдельную функцию `build_*` можно встроить
в эту точку запуска.

Ошибки декодирования, пустого документа и сохранения в примере доходят до вызывающей
стороны. На внешней границе приложения переводи их в принятые в проекте исходы. Если
пустой документ — нормальный случай пропуска, представь это явно в контракте сценария.
Не перехватывай все исключения, чтобы вернуть `False`.

## Как развивать сценарий по шагам

1. Зафиксируй существующий вход, выход и важный побочный эффект поведенческой проверкой,
   если изменение затрагивает существенную логику.
2. Передай создаваемый внутри HTTP/S3/DB-клиент извне, сохраняя текущее поведение.
3. Выдели чистое преобразование и отдели его от загрузки/сохранения.
4. Сделай верхний метод последовательностью смысловых операций; затем убери ненужные обёртки.
5. Добавляй новые этапы и варианты поведения по требованиям задачи.
   Каждое изменение проверяй по результату, а не по форме дерева классов.

## Что проверить

Для приведённого сценария достаточно наблюдаемых инвариантов: успешно сохранённый текст
совпадает с результатом; некорректный вход не сохраняется; ошибка записи не возвращает
успешный результат. Внешние адаптеры проверяй интеграционно согласно их контрактам.
Не привязывай тест к тому, что внутри существует ровно три шага или helper с конкретным именем.
