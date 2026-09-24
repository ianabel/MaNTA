#include "FieldModel.hpp"

// FieldModel itself is header-only. This translation unit exists so that
// FieldModel.o is a real object the link lines can name, and so the registry
// below -- and any out-of-line piece added to it later -- has a home that does
// not force a rebuild of every includer.
//
// The throws below are std::invalid_argument, not std::runtime_error, to match
// PhysicsCases.cpp: this registry mirrors that one, MaNTA.cpp catches the two
// the same way, and a caller distinguishing them would be distinguishing a
// difference that means nothing.

#include <stdexcept>

FieldModels::map_type *FieldModels::map = nullptr;

FieldModels::map_type *FieldModels::getMap()
{
    // Never deleted: this runs during static initialisation and the map must
    // outlive every translation unit that registers into it.
    if (!map)
        map = new map_type;
    return map;
}

void FieldModels::RegisterFieldModel(std::string const &s, function_type creator)
{
    auto *m = getMap();
    if (m->count(s) > 0)
        throw std::invalid_argument("Duplicate field model name '" + s + "'");
    m->insert(std::make_pair(s, creator));
}

std::unique_ptr<FieldModel> FieldModels::InstantiateFieldModel(std::string const &s,
                                                               toml::value const &config,
                                                               Grid const &grid)
{
    auto *m = getMap();
    auto it = m->find(s);
    if (it == m->end())
    {
        std::string known;
        for (auto const &e : *m)
            known += (known.empty() ? "" : ", ") + e.first;
        throw std::invalid_argument("Unknown field model '" + s + "'. Registered field models are: " +
                                    (known.empty() ? "(none)" : known));
    }
    return it->second(config, grid);
}
