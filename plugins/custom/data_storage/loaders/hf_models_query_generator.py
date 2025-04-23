from typing import List

from sqlalchemy import Engine, MetaData, Select, Table, func, select

from embedding_studio.data_storage.loaders.sql.query_generator import (
    QueryGenerator,
)


class HugsearchModelsQueryGenerator(QueryGenerator):
    """
    Query generator for HugSearch models stored in PostgreSQL.
    Builds queries to fetch models and their associated tags.
    """

    def __init__(self, engine: Engine) -> None:
        super().__init__("hf_model", engine)
        self.metadata = MetaData()
        self.table = None
        self.tag_table = None
        self.model_tag_link = None

    def _init_tables(self):
        if self.table is None:
            self.table = Table(
                self.table_name, self.metadata, autoload_with=self.engine
            )
            self.tag_table = Table(
                "tag", self.metadata, autoload_with=self.engine
            )
            self.model_tag_link = Table(
                "hf_model_tag_link", self.metadata, autoload_with=self.engine
            )

    def _coalesce_columns(self, table):
        """Returns a dictionary where text columns are wrapped with COALESCE, keeping original column names."""
        return {
            col.name: func.coalesce(col, "").label(col.name)
            if col.type.python_type == str
            else col
            for col in table.columns
        }

    def fetch_all(self, row_ids: List[str]) -> Select:
        """
        Builds a query to fetch multiple model rows by ID.
        Coalesces NULLs in string columns and aggregates tags.

        :param row_ids: List of model row IDs to retrieve.
        :return: SQL SELECT with model data and aggregated tags.
        """
        self._init_tables()
        coalesced_columns = self._coalesce_columns(self.table)
        return (
            select(
                *coalesced_columns.values(),
                func.coalesce(
                    func.string_agg(
                        func.coalesce(self.tag_table.c.label, ""), ", "
                    ),
                    "",
                ).label("tags"),
            )
            .outerjoin(
                self.model_tag_link,
                self.table.c.id == self.model_tag_link.c.hf_model_id,
            )
            .outerjoin(
                self.tag_table,
                self.model_tag_link.c.tag_id == self.tag_table.c.id,
            )
            .where(self.table.c.id.in_(row_ids))
            .group_by(*coalesced_columns.values())
        )

    def one(self, row_id: str) -> Select:
        """
        Builds a query to fetch a single model row by ID.
        Ensures all text fields are non-null and joins tags.

        :param row_id: ID of the model row to fetch.
        :return: SQL SELECT with one model and its tag list.
        """
        self._init_tables()
        coalesced_columns = self._coalesce_columns(self.table)
        return (
            select(
                *coalesced_columns.values(),
                func.coalesce(
                    func.string_agg(
                        func.coalesce(self.tag_table.c.label, ""), ", "
                    ),
                    "",
                ).label("tags"),
            )
            .outerjoin(
                self.model_tag_link,
                self.table.c.id == self.model_tag_link.c.hf_model_id,
            )
            .outerjoin(
                self.tag_table,
                self.model_tag_link.c.tag_id == self.tag_table.c.id,
            )
            .where(self.table.c.id == row_id)
            .group_by(*coalesced_columns.values())
        )

    def all(self, offset: int, batch_size: int) -> Select:
        """
        Builds a paginated query to fetch model rows.
        Applies COALESCE on text fields and joins tag info.

        :param offset: Number of rows to skip for pagination.
        :param batch_size: Number of rows to fetch per page.
        :return: SQL SELECT with paginated models and tags.
        """
        self._init_tables()
        coalesced_columns = self._coalesce_columns(self.table)
        return (
            select(
                *coalesced_columns.values(),
                func.coalesce(
                    func.string_agg(
                        func.coalesce(self.tag_table.c.label, ""), ", "
                    ),
                    "",
                ).label("tags"),
            )
            .outerjoin(
                self.model_tag_link,
                self.table.c.id == self.model_tag_link.c.hf_model_id,
            )
            .outerjoin(
                self.tag_table,
                self.model_tag_link.c.tag_id == self.tag_table.c.id,
            )
            .group_by(*coalesced_columns.values())
            .order_by(self.table.c.id)
            .limit(batch_size)
            .offset(offset)
        )

    def count(self) -> Select:
        """
        Builds a query to count the number of model rows.

        :return: SQL SELECT returning total number of model rows.
        """
        self._init_tables()
        return super().count()
