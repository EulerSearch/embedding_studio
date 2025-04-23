from typing import List

from sqlalchemy import Engine, MetaData, Select, Table, func, select

from embedding_studio.data_storage.loaders.sql.query_generator import (
    QueryGenerator,
)


class HugsearchCategoriesQueryGenerator(QueryGenerator):
    """
    Query builder for HugSearch category data in PostgreSQL.
    """

    def __init__(self, engine: Engine) -> None:
        super().__init__("category", engine)
        self.metadata = MetaData()
        self.table = None

    def _init_tables(self):
        if self.table is None:
            self.table = Table(
                self.table_name, self.metadata, autoload_with=self.engine
            )

    def fetch_all(self, row_ids: List[str]) -> Select:
        """
        Builds a query to fetch category rows for given row IDs.
        Includes a synthetic field combining intent, tag, and category info.

        :param row_ids: List of string row IDs to retrieve.
        :return: SQL SELECT with full row and synthetic_text column.
        """
        self._init_tables()
        return select(
            self.table,
            func.concat(
                self.table.c.intent,
                "/",
                self.table.c.tag_name,
                "/",
                self.table.c.category_name,
                ":",
                self.table.c.category_value,
            ).label("synthetic_text"),
        ).where(self.table.c.id.in_(row_ids))

    def one(self, row_id: str) -> Select:
        """
        Builds a query to fetch a single category row by ID.
        Includes a synthetic field for embedding generation.

        :param row_id: Unique row ID to fetch.
        :return: SQL SELECT with one row and synthetic_text column.
        """
        self._init_tables()
        return select(
            self.table,
            func.concat(
                self.table.c.intent,
                "/",
                self.table.c.tag_name,
                "/",
                self.table.c.category_name,
                ":",
                self.table.c.category_value,
            ).label("synthetic_text"),
        ).where(self.table.c.id == row_id)

    def all(self, offset: int, batch_size: int) -> Select:
        """
        Builds a paginated query to retrieve category rows.
        Adds synthetic_text for each row and sorts by ID.

        :param offset: Number of rows to skip for pagination.
        :param batch_size: Number of rows to return.
        :return: SQL SELECT with rows and synthetic_text, paginated.
        """
        self._init_tables()
        return (
            select(
                self.table,
                func.concat(
                    self.table.c.intent,
                    "/",
                    self.table.c.tag_name,
                    "/",
                    self.table.c.category_name,
                    ":",
                    self.table.c.category_value,
                ).label("synthetic_text"),
            )
            .order_by(self.table.c.id)
            .limit(batch_size)
            .offset(offset)
        )

    def count(self) -> Select:
        """
        Builds a query to count total number of category rows.

        :return: SQL SELECT with row count from the category table.
        """
        self._init_tables()
        return super(HugsearchCategoriesQueryGenerator, self).count()
