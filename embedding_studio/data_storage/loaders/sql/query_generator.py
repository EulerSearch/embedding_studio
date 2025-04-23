from abc import ABC, abstractmethod
from typing import List

from sqlalchemy import Engine, MetaData, Select, Table, func, select


class AbstractQueryGenerator(ABC):
    """
    Abstract base class for SQL query generators.

    This class defines the interface for creating SQL query generators
    that interact with database tables. It provides abstract methods
    for common database operations.
    """

    def __init__(self, engine: Engine) -> None:
        """
        Initialize the query generator with a database engine.

        :param engine: SQLAlchemy Engine instance for database connection.
        """
        self.engine = engine

    @abstractmethod
    def fetch_all(self, row_ids: List[str]) -> Select:
        """
        Generate a SELECT query to fetch multiple rows by their IDs.

        :param row_ids: List of row identifiers to fetch.
        :return: A SQLAlchemy Select object representing the query.

        Example implementation:
            table = Table('my_table', self.metadata, autoload_with=self.engine)
            return select(table).where(table.c.id.in_(row_ids))
        """

    @abstractmethod
    def one(self, row_id: str) -> Select:
        """
        Generate a SELECT query to fetch a single row by its ID.

        :param row_id: Identifier of the row to fetch.
        :return: A SQLAlchemy Select object representing the query.

        Example implementation:
            table = Table('my_table', self.metadata, autoload_with=self.engine)
            return select(table).where(table.c.id == row_id)
        """

    @abstractmethod
    def all(self, offset: int, batch_size: int) -> Select:
        """
        Generate a SELECT query to fetch a batch of rows with pagination.

        :param offset: Number of rows to skip.
        :param batch_size: Maximum number of rows to return.
        :return: A SQLAlchemy Select object representing the query.

        Example implementation:
            table = Table('my_table', self.metadata, autoload_with=self.engine)
            return select(table).order_by(table.c.id).limit(batch_size).offset(offset)
        """

    @abstractmethod
    def count(self) -> Select:
        """
        Generate a SELECT query to count all rows in the table.

        :return: A SQLAlchemy Select object representing the count query.

        Example implementation:
            table = Table('my_table', self.metadata, autoload_with=self.engine)
            return select(func.count()).select_from(table)
        """


class QueryGenerator(AbstractQueryGenerator):
    """
    Concrete implementation of query generator for SQL databases.

    This class generates SQLAlchemy Select queries for common database
    operations on a specified table.
    """

    def __init__(self, table_name: str, engine: Engine) -> None:
        """
        Initialize the query generator with a table name and database engine.

        :param table_name: Name of the database table to query.
        :param engine: SQLAlchemy Engine instance for database connection.
        """
        super().__init__(engine)
        self.table_name = table_name
        self.metadata = MetaData()  # Removed `bind=self.engine`

    def fetch_all(self, row_ids: List[str]) -> Select:
        """
        Generate a SELECT query to fetch multiple rows by their IDs.

        Creates a query that selects all rows whose ID column value
        is in the provided list of row IDs.

        :param row_ids: List of row identifiers to fetch.
        :return: A SQLAlchemy Select object representing the query.
        """
        table = Table(
            self.table_name, self.metadata, autoload_with=self.engine
        )
        return select(table).where(table.c.id.in_(row_ids))

    def one(self, row_id: str) -> Select:
        """
        Generate a SELECT query to fetch a single row by its ID.

        Creates a query that selects a row whose ID column value
        equals the provided row ID.

        :param row_id: Identifier of the row to fetch.
        :return: A SQLAlchemy Select object representing the query.
        """
        table = Table(
            self.table_name, self.metadata, autoload_with=self.engine
        )
        return select(table).where(table.c.id == row_id)

    def all(self, offset: int, batch_size: int) -> Select:
        """
        Generate a SELECT query to fetch a batch of rows with pagination.

        Creates a query that selects rows with pagination support,
        ordering results by the ID column.

        :param offset: Number of rows to skip.
        :param batch_size: Maximum number of rows to return.
        :return: A SQLAlchemy Select object representing the query.
        """
        table = Table(
            self.table_name, self.metadata, autoload_with=self.engine
        )
        return (
            select(table).order_by(table.c.id).limit(batch_size).offset(offset)
        )

    def count(self) -> Select:
        """
        Generate a SELECT query to count all rows in the table.

        Creates a query that returns the total count of rows in the specified table.

        :return: A SQLAlchemy Select object representing the count query.
        """
        table = Table(
            self.table_name, self.metadata, autoload_with=self.engine
        )
        return select(func.count()).select_from(table)
