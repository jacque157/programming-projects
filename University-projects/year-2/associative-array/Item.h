#ifndef JAKUBOVSKY_PROJEKT_ITEM_H
#define JAKUBOVSKY_PROJEKT_ITEM_H

enum class ItemType { FULL, EMPTY, AVAIL };

template <typename Key, typename Value> class Item
{
    Key identifier;
    Value element;
    ItemType type = ItemType::EMPTY;
public:
    Item() = default;
    Item(const Key key, const Value value) : identifier(key), element(value), type(ItemType::FULL) {};

    Key get_key() { return identifier; };
    Key& key() { return identifier; };
    void set_key(const Key new_key) { identifier = new_key; };

    Value get_value() { return element; };
    Value& value() { return element; };
    void set_value( Value new_value) { element = new_value; };

    void set_type(const ItemType newType) { type = newType; };
    const ItemType get_type() const{ return type; };
    bool is_empty() const { return (type != ItemType::FULL); };
    bool is_full() const { return (type == ItemType::FULL); };
};

#endif //JAKUBOVSKY_PROJEKT_ITEM_H
