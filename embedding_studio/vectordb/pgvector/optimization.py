from embedding_studio.vectordb.optimization import Optimization
from embedding_studio.vectordb.pgvector.collection import Collection


class PgvectorObjectsOptimization(Optimization):
    """
    Base class for PostgreSQL vector database object table optimizations.

    This class provides a framework for implementing optimization strategies
    that can be applied to object tables in pgvector collections.
    """

    def __init__(self, name: str):
        """
        Initialize the objects optimization.

        :param name: A descriptive name for the optimization strategy
        """
        super().__init__(name)

    def _get_statement(self, tablename: str):
        """
        Generate the SQL statement for optimizing the object table.

        This abstract method should be implemented by subclasses to return
        the specific SQL optimization statement for the given table.

        :param tablename: The name of the table to optimize
        :return: SQL statement for optimizing the table
        :raises NotImplementedError: When this method is not implemented in a subclass

        Example implementation:
        ```python
        def _get_statement(self, tablename):
            return sqlalchemy.text(f"VACUUM ANALYZE {tablename}")
        ```
        """
        raise NotImplementedError()

    def __call__(self, collection: Collection):
        """
        Apply the optimization to a collection's object table.

        Executes the SQL statement returned by _get_statement() in a session.

        :param collection: The Collection object to optimize
        """
        with collection.Session() as session:
            stmt = self._get_statement(collection.DbObject.__tablename__)
            session.execute(stmt)
            session.flush()
            session.commit()


class PgvectorObjectPartsOptimization(Optimization):
    """
    Base class for PostgreSQL vector database object parts table optimizations.

    This class provides a framework for implementing optimization strategies
    that can be applied to object parts tables in pgvector collections.
    """

    def __init__(self, name: str):
        """
        Initialize the object parts optimization.

        :param name: A descriptive name for the optimization strategy
        """
        super().__init__(name)

    def _get_statement(self, tablename: str):
        """
        Generate the SQL statement for optimizing the object parts table.

        This abstract method should be implemented by subclasses to return
        the specific SQL optimization statement for the given table.

        :param tablename: The name of the table to optimize
        :return: SQL statement for optimizing the table
        :raises NotImplementedError: When this method is not implemented in a subclass

        Example implementation:
        ```python
        def _get_statement(self, tablename):
            return sqlalchemy.text(f"REINDEX TABLE {tablename}")
        ```
        """
        raise NotImplementedError()

    def __call__(self, collection: Collection):
        """
        Apply the optimization to a collection's object parts table.

        Executes the SQL statement returned by _get_statement() in a session.

        :param collection: The Collection object to optimize
        """
        with collection.Session() as session:
            stmt = self._get_statement(collection.DbObjectPart.__tablename__)
            session.execute(stmt)
            session.flush()
            session.commit()
