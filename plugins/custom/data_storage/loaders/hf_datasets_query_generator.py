from typing import List

from sqlalchemy import Engine, MetaData, Select, Table, func, select

from embedding_studio.data_storage.loaders.sql.query_generator import (
    QueryGenerator,
)


class HugsearchDatasetsQueryGenerator(QueryGenerator):
    """
    Query generator for HugSearch datasets stored in PostgreSQL.
    Builds queries to fetch datasets and their associated tags.
    """

    def __init__(self, engine: Engine) -> None:
        super().__init__("hf_dataset", engine)
        self.metadata = MetaData()  # Removed `bind=self.engine`

        self.table = None
        self.tag_table = None
        self.dataset_tag_link = None

    def _init_tables(self):
        if self.table is None:
            # Use autoload_with=self.engine when defining tables
            self.table = Table(
                self.table_name, self.metadata, autoload_with=self.engine
            )
            self.tag_table = Table(
                "tag_dataset", self.metadata, autoload_with=self.engine
            )
            self.dataset_tag_link = Table(
                "hf_dataset_tag_link", self.metadata, autoload_with=self.engine
            )

    def fetch_all(self, row_ids: List[str]) -> Select:
        """
        Builds a query to fetch multiple dataset rows by ID.
        Joins dataset_tag_link and tag_dataset to aggregate tags.

        :param row_ids: List of dataset row IDs to fetch.
        :return: SQL SELECT with dataset rows and aggregated tags.
        """
        self._init_tables()
        return (
            select(
                self.table,
                func.string_agg(self.tag_table.c.label, ", ").label("tags"),
            )
            .join(
                self.dataset_tag_link,
                self.table.c.id == self.dataset_tag_link.c.hf_dataset_id,
            )
            .join(
                self.tag_table,
                self.dataset_tag_link.c.tag_id == self.tag_table.c.id,
            )
            .where(self.table.c.id.in_(row_ids))
            .group_by(self.table.c.id)
        )

    def one(self, row_id: str) -> Select:
        """
        Builds a query to fetch a single dataset row by ID.
        Includes tags as a comma-separated string.

        :param row_id: ID of the dataset to retrieve.
        :return: SQL SELECT with one dataset and its tags.
        """
        self._init_tables()
        return (
            select(
                self.table,
                func.string_agg(self.tag_table.c.label, ", ").label("tags"),
            )
            .join(
                self.dataset_tag_link,
                self.table.c.id == self.dataset_tag_link.c.hf_dataset_id,
            )
            .join(
                self.tag_table,
                self.dataset_tag_link.c.tag_id == self.tag_table.c.id,
            )
            .where(self.table.c.id == row_id)
            .group_by(self.table.c.id)
        )

    def all(self, offset: int, batch_size: int) -> Select:
        """
        Builds a paginated query to fetch dataset rows.
        Joins tags and groups results by dataset ID.

        :param offset: Number of rows to skip (for pagination).
        :param batch_size: Number of rows to fetch.
        :return: SQL SELECT with datasets and tags, paginated.
        """
        self._init_tables()
        return (
            select(
                self.table,
                func.string_agg(self.tag_table.c.label, ", ").label("tags"),
            )
            .join(
                self.dataset_tag_link,
                self.table.c.id == self.dataset_tag_link.c.hf_dataset_id,
            )
            .join(
                self.tag_table,
                self.dataset_tag_link.c.tag_id == self.tag_table.c.id,
            )
            .group_by(self.table.c.id)
            .order_by(self.table.c.id)
            .limit(batch_size)
            .offset(offset)
        )

    def count(self) -> Select:
        """
        Builds a query to count all dataset rows.

        :return: SQL SELECT that returns dataset row count.
        """
        self._init_tables()
        return super(HugsearchDatasetsQueryGenerator, self).count()
