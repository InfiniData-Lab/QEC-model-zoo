class StringUtils:
    @staticmethod
    def normalize_string(string: str) -> str:
        """
        Normalize string to lower case and replace spaces with _.

        :param string: String to be normalized.
        :return: Normalized string.
        """
        return string.lower().replace(" ", "_").strip()

    @staticmethod
    def filter_by_match(objects, names, *, attribute=None):
        """
        Filter items by case-insensitive partial string matching.

        If `attribute` is provided, matches against that attribute of each object.
        Otherwise, assumes the items are strings.

        Args:
            objects (iterable): Strings or objects to filter.
            names (iterable[str]): Strings to search for.
            attribute (str, optional): Attribute to match against.
                Defaults to None, treating items as strings.

        Returns:
            list: Items containing at least one matching substring.

        Examples:
            >>> filter_by_match(["Alice Smith", "Bob", "Charlie"], ["alice", "bob"])
            ['Alice Smith', 'Bob']

            >>> filter_by_match(users, ["alice", "bob"], attribute="name")
            [<User Alice Smith>, <User Bobby>]
        """
        names = [name.casefold() for name in names]

        return [
            obj for obj in objects
            if any(
                name in str(
                    getattr(obj, attribute, None) if attribute else obj
                ).casefold()
                for name in names
            )
        ]
