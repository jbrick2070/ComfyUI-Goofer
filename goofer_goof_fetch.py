"""
GooferGoofFetch — Fetches movie metadata and goofs using Cinemagoer.

No API keys needed. Cinemagoer (formerly IMDbPY) is MIT-licensed and
queries IMDb's public data without scraping HTML.

Outputs:
  - movie_data: dict with title, year, genres, plot, imdb_id
  - goofs_data: list of goof dicts [{category, description}, ...]

Fallback: if Cinemagoer fails or movie not found, returns empty goofs
with a placeholder so downstream nodes can still generate content.

Author: Jeffrey A. Brick
"""

import hashlib
import json
import logging
import os
import random
import re
import time

log = logging.getLogger("Goofer.GoofFetch")

# ── Disk cache ────────────────────────────────────────────────────────
# Avoids hitting Cinemagoer on every queue run.  Cache lives in
# ComfyUI's temp directory and is keyed on (title, year).
_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".goofer_cache")


def _cache_key(title: str, year: int) -> str:
    slug = f"{title.lower().strip()}_{year}"
    return hashlib.sha256(slug.encode()).hexdigest()[:16]


def _cache_get(title: str, year: int):
    """Return (movie_data, goofs_list, cast_names) from cache, or None."""
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        path = os.path.join(_CACHE_DIR, f"{_cache_key(title, year)}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            log.info("[GoofFetch] Cache HIT for '%s' (%d)", title, year)
            return data["movie_data"], data["goofs"], data.get("cast_names", [])
    except Exception as e:
        log.debug("[GoofFetch] Cache read error: %s", e)
    return None


def _cache_put(title: str, year: int, movie_data: dict, goofs: list,
               cast_names: list):
    """Write fetch results to disk cache."""
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        path = os.path.join(_CACHE_DIR, f"{_cache_key(title, year)}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"movie_data": movie_data, "goofs": goofs,
                        "cast_names": cast_names}, f, ensure_ascii=False)
        log.info("[GoofFetch] Cache WRITE for '%s' (%d)", title, year)
    except Exception as e:
        log.debug("[GoofFetch] Cache write error: %s", e)


# ── Curated movie pool ────────────────────────────────────────────────
# Movies verified to have 3+ goofs on IMDb. Used when random_movie=True.
# Titles must match IMDb exactly to avoid Cinemagoer search misses.

# Each entry: (title, year, imdb_id)
# imdb_id enables direct get_movie() lookup as fallback if title search fails.
_RANDOM_MOVIE_POOL = [
    # Action / Thriller
    ("Die Hard",                              1988, "0095016"),
    ("Lethal Weapon",                         1987, "0093409"),
    ("Speed",                                 1994, "0111257"),
    ("The Rock",                              1996, "0117500"),
    ("Con Air",                               1997, "0118583"),
    ("Air Force One",                         1997, "0118571"),
    ("Armageddon",                            1998, "0120591"),
    ("The Dark Knight",                       2008, "0468569"),
    ("Mission: Impossible - Fallout",         2018, "4912910"),
    ("Top Gun",                               1986, "0092099"),
    ("Point Break",                           1991, "0102685"),
    ("Total Recall",                          1990, "0100802"),
    ("RoboCop",                               1987, "0093870"),
    # Sci-Fi
    ("Jurassic Park",                         1993, "0107290"),
    ("The Matrix",                            1999, "0133093"),
    ("Terminator 2: Judgment Day",            1991, "0103064"),
    ("Alien",                                 1979, "0078748"),
    ("Aliens",                                1986, "0090605"),
    ("Predator",                              1987, "0093773"),
    ("Back to the Future",                    1985, "0088763"),
    ("Blade Runner",                          1982, "0083658"),
    ("Interstellar",                          2014, "0816692"),
    ("Gravity",                               2013, "1454468"),
    ("Avatar",                                2009, "0499549"),
    ("Independence Day",                      1996, "0116629"),
    # Drama / Crime
    ("The Godfather",                         1972, "0068646"),
    ("Goodfellas",                            1990, "0099685"),
    ("Pulp Fiction",                          1994, "0110912"),
    ("The Shawshank Redemption",              1994, "0111161"),
    ("Fight Club",                            1999, "0137523"),
    ("Heat",                                  1995, "0113277"),
    ("Se7en",                                 1995, "0114369"),
    # Horror
    ("The Shining",                           1980, "0081505"),
    ("Jaws",                                  1975, "0073195"),
    ("A Nightmare on Elm Street",             1984, "0087800"),
    ("Halloween",                             1978, "0077651"),
    ("The Silence of the Lambs",              1991, "0102926"),
    ("Poltergeist",                           1982, "0084516"),
    # Adventure
    ("Raiders of the Lost Ark",               1981, "0082971"),
    ("Ghostbusters",                          1984, "0087332"),
    ("E.T. the Extra-Terrestrial",            1982, "0083866"),
    ("Ferris Bueller's Day Off",              1986, "0091042"),
    ("Forrest Gump",                          1994, "0109830"),
    ("Braveheart",                            1995, "0112573"),
    ("Gladiator",                             2000, "0172495"),
    ("Troy",                                  2004, "0332452"),
    # Superhero / Fantasy
    ("Iron Man",                              2008, "0371746"),
    ("Thor",                                  2011, "0800369"),
    ("Captain America: The First Avenger",    2011, "0458339"),
    ("Man of Steel",                          2013, "0770828"),
    ("Inception",                             2010, "1375666"),
    # ── High goof count — these are IMDb goof legends ──────────────────
    # Commando arguably has more documented goofs than any film on IMDb
    ("Commando",                              1985, "0088944"),
    ("Navy SEALs",                            1990, "0100232"),
    ("Under Siege",                           1992, "0105690"),
    ("Hard Target",                           1993, "0107698"),
    ("Demolition Man",                        1993, "0106697"),
    ("Cliffhanger",                           1993, "0106582"),
    ("Face/Off",                              1997, "0119094"),
    ("Broken Arrow",                          1996, "0115759"),
    ("Eraser",                                1996, "0116213"),
    ("The Last Boy Scout",                    1991, "0102266"),
    ("Highlander",                            1986, "0091203"),
    ("Conan the Barbarian",                   1982, "0082198"),
    ("Pearl Harbor",                          2001, "0213149"),
    ("The Patriot",                           2000, "0187393"),
    ("Kingdom of Heaven",                     2005, "0320661"),
    ("Alexander",                             2004, "0346491"),
    ("Transformers: Revenge of the Fallen",   2009, "1055369"),
    ("Fair Game",                             1995, "0113029"),
    # ── Superhero / MCU / DC ──────────────────────────────────────────────
    ("The Avengers",                          2012, "0848228"),
    ("Avengers: Infinity War",                2018, "4154756"),
    ("Avengers: Endgame",                     2019, "4154796"),
    ("Captain America: Civil War",            2016, "3498820"),
    ("Spider-Man: Homecoming",                2017, "2250912"),
    ("Spider-Man: No Way Home",               2021, "10872600"),
    ("Spider-Man: Far from Home",             2019, "6320628"),
    ("Black Panther",                         2018, "1825683"),
    ("Doctor Strange",                        2016, "1211837"),
    ("Guardians of the Galaxy",               2014, "2015381"),
    ("Thor: Ragnarok",                        2017, "3501632"),
    ("Ant-Man",                               2015, "0478970"),
    ("Captain Marvel",                        2019, "4154664"),
    ("Batman Begins",                         2005, "0372784"),
    ("The Dark Knight Rises",                 2012, "1345836"),
    ("Wonder Woman",                          2017, "0451279"),
    ("Justice League",                        2017, "0974015"),
    ("Deadpool",                              2016, "1431045"),
    ("X-Men: Days of Future Past",            2014, "1877832"),
    ("Superman Returns",                      2006, "0348150"),
    ("The Incredible Hulk",                   2008, "0800080"),
    # ── Action / Thriller (additional) ────────────────────────────────────
    ("John Wick",                             2014, "2911666"),
    ("Mission: Impossible - Rogue Nation",    2015, "2381249"),
    ("Kingsman: The Secret Service",          2014, "2802144"),
    ("Edge of Tomorrow",                      2014, "1631867"),
    ("Pacific Rim",                           2013, "1663662"),
    ("Olympus Has Fallen",                    2013, "2302755"),
    ("White House Down",                      2013, "2334879"),
    # ── Adventure / Fantasy ───────────────────────────────────────────────
    ("Pirates of the Caribbean: The Curse of the Black Pearl", 2003, "0325980"),
    ("Pirates of the Caribbean: Dead Man's Chest",             2006, "0383574"),
    ("Pirates of the Caribbean: At World's End",               2007, "0449088"),
    ("The Mummy",                             1999, "0120616"),
    ("The Mummy Returns",                     2001, "0209163"),
    ("National Treasure",                     2004, "0368891"),
    ("Oblivion",                              2013, "1483013"),
    # ── Sci-Fi / Monster (additional) ─────────────────────────────────────
    ("Godzilla",                              2014, "0831387"),
    ("Transformers",                          2007, "0418279"),
    ("Transformers: Dark of the Moon",        2011, "1399103"),
    ("Battleship",                            2012, "1440129"),
    ("Real Steel",                            2011, "0433035"),
    # ── Classic / Iconic Goof-Heavy ───────────────────────────────────────
    ("The Wizard of Oz",                      1939, "0032138"),
    ("Ben-Hur",                               1959, "0052618"),
    ("Spartacus",                             1960, "0054331"),
    ("North by Northwest",                    1959, "0053125"),
    ("Dirty Harry",                           1971, "0066999"),
    ("Bullitt",                               1968, "0062765"),
    ("The French Connection",                 1971, "0067116"),
    ("Scarface",                              1983, "0086250"),
    ("First Blood",                           1982, "0083944"),
    ("Rambo: First Blood Part II",            1985, "0089880"),
    ("Rocky",                                 1976, "0075148"),
    ("Rocky IV",                              1985, "0089927"),
    ("The Terminator",                        1984, "0088247"),
    ("Escape from New York",                  1981, "0082340"),
    ("Big Trouble in Little China",           1986, "0090728"),
    ("They Live",                             1988, "0096256"),
    ("Tango & Cash",                          1989, "0098439"),
    ("Cobra",                                 1986, "0090859"),
    ("Bloodsport",                            1988, "0092675"),
    ("Kickboxer",                             1989, "0097659"),
    ("Enter the Dragon",                      1973, "0070034"),
    ("The Fugitive",                          1993, "0106977"),
    ("Twister",                               1996, "0117998"),
    ("Dante's Peak",                          1997, "0118928"),
    ("Volcano",                               1997, "0120461"),
    ("Waterworld",                            1995, "0114898"),
    ("The Postman",                           1997, "0119925"),
    ("Starship Troopers",                     1997, "0120201"),
    ("Men in Black",                          1997, "0119654"),
    ("The Fifth Element",                     1997, "0119116"),
    ("Titanic",                               1997, "0120338"),
    ("Saving Private Ryan",                   1998, "0120815"),
    ("Cast Away",                             2000, "0162222"),
    ("The Lord of the Rings: The Fellowship of the Ring",  2001, "0120737"),
    ("The Lord of the Rings: The Two Towers",              2002, "0167261"),
    ("The Lord of the Rings: The Return of the King",      2003, "0167260"),
    ("Harry Potter and the Sorcerer's Stone",              2001, "0241527"),
    ("Harry Potter and the Chamber of Secrets",            2002, "0295297"),
    ("Harry Potter and the Prisoner of Azkaban",           2004, "0304141"),
    ("Harry Potter and the Goblet of Fire",                2005, "0330373"),
    ("Star Wars: Episode IV - A New Hope",    1977, "0076759"),
    ("Star Wars: Episode V - The Empire Strikes Back", 1980, "0080684"),
    ("Star Wars: Episode VI - Return of the Jedi",     1983, "0086190"),
    ("Star Wars: Episode I - The Phantom Menace",      1999, "0120915"),
    ("Star Wars: Episode II - Attack of the Clones",   2002, "0121765"),
    ("Star Wars: Episode III - Revenge of the Sith",   2005, "0121766"),
    ("Mad Max: Fury Road",                    2015, "1392190"),
    ("Dunkirk",                               2017, "5013056"),
    ("1917",                                  2019, "8579674"),
    ("Tenet",                                 2020, "6723592"),
    ("No Country for Old Men",                2007, "0477348"),
    ("There Will Be Blood",                   2007, "0469494"),
    ("The Departed",                          2006, "0407887"),
    ("Casino Royale",                         2006, "0381061"),
    ("Skyfall",                               2012, "1074638"),
    ("GoldenEye",                             1995, "0113189"),
    ("The Bourne Identity",                   2002, "0258463"),
    ("The Bourne Ultimatum",                  2007, "0440963"),
    ("Kill Bill: Volume 1",                   2003, "0266697"),
    ("Inglourious Basterds",                  2009, "0361748"),
    ("Django Unchained",                      2012, "1853728"),
    ("The Revenant",                          2015, "1663202"),
    ("Fury",                                  2014, "2713180"),
    ("Hacksaw Ridge",                         2016, "2119532"),
    ("Apocalypse Now",                        1979, "0078788"),
    ("Platoon",                               1986, "0091763"),
    ("Full Metal Jacket",                     1987, "0093058"),
    ("Black Hawk Down",                       2001, "0265086"),
    # ── Comedy ────────────────────────────────────────────────────────────
    ("Airplane!",                             1980, "0080339"),
    ("The Naked Gun: From the Files of Police Squad!", 1988, "0095705"),
    ("Blazing Saddles",                       1974, "0071230"),
    ("Caddyshack",                            1980, "0080487"),
    ("National Lampoon's Vacation",           1983, "0085995"),
    ("Animal House",                          1978, "0077975"),
    ("The Blues Brothers",                    1980, "0080455"),
    ("Beverly Hills Cop",                     1984, "0086960"),
    ("Trading Places",                        1983, "0086465"),
    ("Coming to America",                     1988, "0094898"),
    ("Groundhog Day",                         1993, "0107048"),
    ("Mrs. Doubtfire",                        1993, "0107614"),
    ("Ace Ventura: Pet Detective",            1994, "0109040"),
    ("Dumb and Dumber",                       1994, "0109686"),
    ("Austin Powers: International Man of Mystery", 1997, "0118655"),
    ("There's Something About Mary",          1998, "0129387"),
    ("Office Space",                          1999, "0151804"),
    ("Superbad",                              2007, "0829482"),
    ("The Hangover",                          2009, "1119646"),
    ("Bridesmaids",                           2011, "1478338"),
    ("Step Brothers",                         2008, "0838283"),
    ("Anchorman: The Legend of Ron Burgundy",  2004, "0357413"),
    ("Zoolander",                             2001, "0196229"),
    ("Dodgeball: A True Underdog Story",      2004, "0364725"),
    ("Shaun of the Dead",                     2004, "0365748"),
    ("Hot Fuzz",                              2007, "0425112"),
    # ── Horror / Thriller (additional) ────────────────────────────────────
    ("The Exorcist",                          1973, "0070047"),
    ("The Thing",                             1982, "0084787"),
    ("An American Werewolf in London",        1981, "0082010"),
    ("The Evil Dead",                         1981, "0083907"),
    ("Evil Dead II",                          1987, "0092991"),
    ("Psycho",                                1960, "0054215"),
    ("The Omen",                              1976, "0075005"),
    ("Scream",                                1996, "0117571"),
    ("The Blair Witch Project",               1999, "0185937"),
    ("It",                                    2017, "1396484"),
    ("Get Out",                               2017, "5052448"),
    ("A Quiet Place",                         2018, "6644200"),
    ("Hereditary",                            2018, "7784604"),
    ("The Conjuring",                         2013, "1457767"),
    ("Saw",                                   2004, "0387564"),
    ("28 Days Later",                         2002, "0289043"),
    ("Friday the 13th",                       1980, "0080761"),
    ("The Texas Chain Saw Massacre",          1974, "0072271"),
    ("Hellraiser",                            1987, "0093177"),
    ("Child's Play",                          1988, "0094862"),
    # ── Drama / Crime (additional) ────────────────────────────────────────
    ("Casino",                                1995, "0112641"),
    ("The Untouchables",                      1987, "0094226"),
    ("American Beauty",                       1999, "0169547"),
    ("A Beautiful Mind",                      2001, "0268978"),
    ("Schindler's List",                      1993, "0108052"),
    ("The Green Mile",                        1999, "0120689"),
    ("American History X",                    1998, "0120586"),
    ("Crash",                                 2004, "0375679"),
    ("Million Dollar Baby",                   2004, "0405159"),
    ("Mystic River",                          2003, "0327056"),
    ("Gone Girl",                             2014, "2267998"),
    ("Zodiac",                                2007, "0443706"),
    ("Prisoners",                             2013, "1392214"),
    ("Sicario",                               2015, "3397884"),
    ("The Town",                              2010, "0840361"),
    ("The Wolf of Wall Street",               2013, "0993846"),
    ("The Big Short",                         2015, "1596363"),
    ("Moneyball",                             2011, "1210166"),
    ("The Social Network",                    2010, "1285016"),
    # ── Classic Hitchcock / Noir ──────────────────────────────────────────
    ("Rear Window",                           1954, "0047396"),
    ("Vertigo",                               1958, "0052357"),
    ("The Birds",                             1963, "0056869"),
    ("Dial M for Murder",                     1954, "0046912"),
    # ── Western ───────────────────────────────────────────────────────────
    ("The Good, the Bad and the Ugly",        1966, "0060196"),
    ("Unforgiven",                            1992, "0105695"),
    ("Tombstone",                             1993, "0108358"),
    ("Wyatt Earp",                            1994, "0111756"),
    ("Butch Cassidy and the Sundance Kid",    1969, "0064115"),
    ("True Grit",                             2010, "1403865"),
    ("The Magnificent Seven",                 1960, "0054047"),
    ("3:10 to Yuma",                          2007, "0381849"),
    ("Open Range",                            2003, "0316356"),
    ("Dances with Wolves",                    1990, "0099348"),
    # ── Disaster ──────────────────────────────────────────────────────────
    ("The Towering Inferno",                  1974, "0072308"),
    ("The Poseidon Adventure",                1972, "0069113"),
    ("Earthquake",                            1974, "0071455"),
    ("Deep Impact",                           1998, "0120647"),
    ("The Day After Tomorrow",                2004, "0319262"),
    ("2012",                                  2009, "1190080"),
    ("San Andreas",                           2015, "2126355"),
    ("Geostorm",                              2017, "1981128"),
    # ── Spy / Espionage ───────────────────────────────────────────────────
    ("The Spy Who Loved Me",                  1977, "0076752"),
    ("Moonraker",                             1979, "0079574"),
    ("Tomorrow Never Dies",                   1997, "0120347"),
    ("The World Is Not Enough",               1999, "0143145"),
    ("Die Another Day",                       2002, "0246460"),
    ("Spectre",                               2015, "2379713"),
    ("No Time to Die",                        2021, "2382320"),
    ("Mission: Impossible",                   1996, "0117060"),
    ("Mission: Impossible II",                2000, "0120755"),
    ("Mission: Impossible III",               2006, "0317919"),
    ("Mission: Impossible - Ghost Protocol",  2011, "1229238"),
    ("The Bourne Supremacy",                  2004, "0372183"),
    ("Salt",                                  2010, "0944835"),
    ("Atomic Blonde",                         2017, "2406566"),
    ("Red Sparrow",                           2018, "2873282"),
    # ── Sci-Fi (additional) ───────────────────────────────────────────────
    ("The Day the Earth Stood Still",         1951, "0043456"),
    ("2001: A Space Odyssey",                 1968, "0062622"),
    ("Close Encounters of the Third Kind",    1977, "0075860"),
    ("War of the Worlds",                     2005, "0407304"),
    ("I, Robot",                              2004, "0343818"),
    ("Minority Report",                       2002, "0181689"),
    ("A.I. Artificial Intelligence",          2001, "0212720"),
    ("District 9",                            2009, "1136608"),
    ("Arrival",                               2016, "2543164"),
    ("Ex Machina",                            2014, "0470752"),
    ("The Martian",                           2015, "3659388"),
    ("Prometheus",                            2012, "1446714"),
    ("Alien: Covenant",                       2017, "2316204"),
    ("Blade Runner 2049",                     2017, "1856101"),
    ("Elysium",                               2013, "1535108"),
    ("Looper",                                2012, "1276104"),
    ("Snowpiercer",                           2013, "1706620"),
    ("The Hunger Games",                      2012, "1392170"),
    ("The Hunger Games: Catching Fire",       2013, "1951264"),
    ("Divergent",                             2014, "1840309"),
    ("The Maze Runner",                       2014, "1790864"),
    ("Ender's Game",                          2013, "1731141"),
    ("Ready Player One",                      2018, "1677720"),
    # ── Animation / Family ────────────────────────────────────────────────
    ("Toy Story",                             1995, "0114709"),
    ("Toy Story 2",                           1999, "0120363"),
    ("Toy Story 3",                           2010, "0435761"),
    ("Finding Nemo",                          2003, "0266543"),
    ("The Lion King",                         1994, "0110357"),
    ("Shrek",                                 2001, "0126029"),
    ("Shrek 2",                               2004, "0298148"),
    ("Monsters, Inc.",                        2001, "0198781"),
    ("The Incredibles",                       2004, "0317705"),
    ("Up",                                    2009, "1049413"),
    ("WALL-E",                                2008, "0910970"),
    ("Frozen",                                2013, "2294629"),
    ("Zootopia",                              2016, "2948356"),
    ("Coco",                                  2017, "2380307"),
    ("Ratatouille",                           2007, "0382932"),
    ("Inside Out",                            2015, "2096673"),
    ("Beauty and the Beast",                  1991, "0101414"),
    ("Aladdin",                               1992, "0103639"),
    ("The Little Mermaid",                    1989, "0097757"),
    # ── Sports ────────────────────────────────────────────────────────────
    ("Raging Bull",                           1980, "0081398"),
    ("Rocky II",                              1979, "0079817"),
    ("Rocky III",                             1982, "0084602"),
    ("Creed",                                 2015, "3076658"),
    ("Remember the Titans",                   2000, "0210945"),
    ("Hoosiers",                              1986, "0091217"),
    ("Field of Dreams",                       1989, "0097351"),
    ("Rudy",                                  1993, "0108002"),
    ("The Longest Yard",                      2005, "0398165"),
    # ── Romance / Drama ───────────────────────────────────────────────────
    ("Pretty Woman",                          1990, "0100405"),
    ("The Notebook",                          2004, "0332280"),
    ("Ghost",                                 1990, "0099653"),
    ("Dirty Dancing",                         1987, "0092890"),
    ("Grease",                                1978, "0077631"),
    ("Sleepless in Seattle",                  1993, "0108160"),
    ("When Harry Met Sally...",               1989, "0098635"),
    # ── Musical ───────────────────────────────────────────────────────────
    ("Singin' in the Rain",                   1952, "0045152"),
    ("The Sound of Music",                    1965, "0059742"),
    ("West Side Story",                       1961, "0055614"),
    ("Chicago",                               2002, "0299658"),
    ("Les Miserables",                        2012, "1707386"),
    ("La La Land",                            2016, "3783958"),
    ("The Greatest Showman",                  2017, "1485796"),
    # ── War (additional) ──────────────────────────────────────────────────
    ("The Bridge on the River Kwai",          1957, "0050212"),
    ("The Great Escape",                      1963, "0057115"),
    ("Patton",                                1970, "0066206"),
    ("A Bridge Too Far",                      1977, "0075784"),
    ("The Dirty Dozen",                       1967, "0061578"),
    ("Where Eagles Dare",                     1968, "0065207"),
    ("The Guns of Navarone",                  1961, "0054953"),
    ("Midway",                                2019, "6924650"),
    ("Lone Survivor",                         2013, "1091191"),
    ("American Sniper",                       2014, "2179136"),
    ("Jarhead",                               2005, "0418763"),
    ("We Were Soldiers",                      2002, "0277434"),
    ("Letters from Iwo Jima",                 2006, "0498380"),
    ("Flags of Our Fathers",                  2006, "0418689"),
    # ── 80s/90s Action Legends ────────────────────────────────────────────
    ("Predator 2",                            1990, "0100403"),
    ("Escape from L.A.",                      1996, "0116225"),
    ("The Running Man",                       1987, "0093894"),
    ("Kindergarten Cop",                      1990, "0099938"),
    ("Last Action Hero",                      1993, "0107362"),
    ("True Lies",                             1994, "0111503"),
    ("The Long Kiss Goodnight",               1996, "0116908"),
    ("Executive Decision",                    1996, "0116253"),
    ("Sudden Death",                          1995, "0114576"),
    ("Passenger 57",                          1992, "0105104"),
    ("Marked for Death",                      1990, "0100114"),
    ("Above the Law",                         1988, "0094602"),
    ("Hard to Kill",                          1990, "0099739"),
    ("Out for Justice",                       1991, "0102613"),
    ("On Deadly Ground",                      1994, "0110725"),
    ("Universal Soldier",                     1992, "0105698"),
    ("Timecop",                               1994, "0111438"),
    ("Double Impact",                         1991, "0101764"),
    ("Lionheart",                             1990, "0099964"),
    ("Missing in Action",                     1984, "0087727"),
    ("Delta Force",                           1986, "0090927"),
    ("Invasion U.S.A.",                       1985, "0089345"),
    ("Red Scorpion",                          1988, "0098188"),
    ("Cyborg",                                1989, "0097138"),
    ("Road House",                            1989, "0098206"),
    ("Lock Up",                               1989, "0097770"),
    ("Over the Top",                          1987, "0093692"),
    ("Red Heat",                              1988, "0095963"),
    ("Raw Deal",                              1986, "0091828"),
    ("Showdown in Little Tokyo",              1991, "0102915"),
    ("Stone Cold",                            1991, "0102984"),
    ("The Specialist",                        1994, "0111359"),
    ("The Expendables",                       2010, "1320253"),
    ("Rambo III",                             1988, "0095956"),
    ("Rambo",                                 2008, "0462499"),
    ("Die Hard 2",                            1990, "0099423"),
    ("Die Hard with a Vengeance",             1995, "0112864"),
    ("Lethal Weapon 2",                       1989, "0097733"),
    ("Lethal Weapon 3",                       1992, "0104714"),
    ("Lethal Weapon 4",                       1998, "0122151"),
    # ── Recent Blockbusters (2020s) ───────────────────────────────────────
    ("Dune",                                  2021, "1160419"),
    ("Dune: Part Two",                        2024, "15239678"),
    ("Top Gun: Maverick",                     2022, "1745960"),
    ("The Batman",                            2022, "1877830"),
    ("Everything Everywhere All at Once",     2022, "6710474"),
    ("Oppenheimer",                           2023, "15398776"),
    ("Barbie",                                2023, "1517268"),
    ("John Wick: Chapter 4",                  2023, "10366206"),
    ("Knives Out",                            2019, "8946378"),
    ("Glass Onion",                           2022, "11564570"),
    ("Free Guy",                              2021, "6264654"),
    ("Shang-Chi and the Legend of the Ten Rings", 2021, "9376612"),
    ("Eternals",                              2021, "9032400"),
    # ── Cult / Genre Classics ─────────────────────────────────────────────
    ("Reservoir Dogs",                        1992, "0105236"),
    ("From Dusk Till Dawn",                   1996, "0116367"),
    ("Sin City",                              2005, "0401792"),
    ("300",                                   2006, "0416449"),
    ("Watchmen",                              2009, "0409459"),
    ("V for Vendetta",                        2005, "0434409"),
    ("The Crow",                              1994, "0109506"),
    ("Stargate",                              1994, "0111282"),
    ("Total Recall",                          2012, "1386703"),
    ("Dredd",                                 2012, "1343727"),
    ("Judge Dredd",                           1995, "0113492"),
    ("Mortal Kombat",                         1995, "0113855"),
    ("Street Fighter",                        1994, "0111301"),
]

# Quick lookup: title+year → imdb_id (for fallback get_movie())
_IMDB_ID_MAP = {
    (t.lower(), y): iid for t, y, iid in _RANDOM_MOVIE_POOL
}


# ── Goof category mapping ─────────────────────────────────────────────
# Cinemagoer returns goof categories as keys in the goofs dict.
# We normalize them to clean display names.
_CATEGORY_MAP = {
    "continuity":       "Continuity",
    "factual error":    "Factual Error",
    "factual errors":   "Factual Error",
    "revealing mistake": "Revealing Mistake",
    "revealing mistakes": "Revealing Mistake",
    "audio/visual unsynchronized": "Audio/Visual Unsync",
    "audio/visual unsynchronised": "Audio/Visual Unsync",
    "anachronism":      "Anachronism",
    "anachronisms":     "Anachronism",
    "crew or equipment visible": "Crew Visible",
    "crew/equipment visible": "Crew Visible",
    "boom mic visible": "Boom Mic Visible",
    "plot hole":        "Plot Hole",
    "plot holes":       "Plot Hole",
    "character error":  "Character Error",
    "character errors":  "Character Error",
    "miscellaneous":    "Miscellaneous",
    "errors in geography": "Geography Error",
    "geographical errors": "Geography Error",
}


def _normalize_category(raw: str) -> str:
    """Normalize a raw goof category string."""
    key = raw.strip().lower()
    return _CATEGORY_MAP.get(key, raw.strip().title())


class GooferGoofFetch:
    """Fetches movie goofs via Cinemagoer (no API key needed).

    Also extracts top-billed cast + character names from IMDb and
    passes them as ``cast_names`` inside movie_data so the downstream
    sanitizer can strip movie-specific names automatically.
    """

    CATEGORY = "Goofer"
    FUNCTION = "fetch"
    RETURN_TYPES = ("GOOFER_MOVIE", "GOOFER_GOOFS",)
    RETURN_NAMES = ("movie_data", "goofs_data",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("GOOFER_CONFIG",),
            },
            "optional": {
                "manual_goofs_json": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": (
                        "Optional: paste a JSON array of goofs manually. "
                        "Format: [{\"category\": \"Continuity\", \"description\": \"...\"}]. "
                        "If provided, skips Cinemagoer fetch entirely."
                    )
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time()

    def fetch(self, config, manual_goofs_json=""):
        max_goofs = config["max_goofs"]
        seed = config.get("seed", None)

        # ── Random movie selection ──
        if config.get("random_movie", False):
            rng_movie = random.Random(int(time.time_ns()))
            entry = rng_movie.choice(_RANDOM_MOVIE_POOL)
            title, year = entry[0], entry[1]
            log.info("[GoofFetch] Random movie: '%s' (%d)", title, year)
            # When random_movie is on, also randomize which goofs are picked
            # so every run draws a genuinely different set from the pool.
            seed = int(time.time_ns()) % (2 ** 31)
        else:
            title = config["movie_title"]
            year = config["movie_year"]

        # ── Manual override: skip fetch if user provides JSON ──
        if manual_goofs_json.strip():
            movie_data, goofs = self._parse_manual(manual_goofs_json, title, year)
            goofs = goofs[:max_goofs]
            log.info("[GoofFetch] Manual input: %d goofs for '%s'", len(goofs), title)
            return (movie_data, goofs)

        # ── Check disk cache first ──
        cached = _cache_get(title, year)
        if cached is not None:
            movie_data, goofs, cast_names = cached
            movie_data["cast_names"] = cast_names
            # Diversify from the full cached pool using the current seed
            # so every run with a different seed draws a fresh set of goofs.
            goofs = self._diversify(goofs, max_goofs, seed=seed)
            if not goofs:
                goofs = self._placeholder_goofs(movie_data, max_goofs)
            log.info("[GoofFetch] Cache HIT → %d goofs (seed=%s)", len(goofs), seed)
            return (movie_data, goofs)

        # ── Cinemagoer fetch ──
        try:
            from imdb import Cinemagoer
        except ImportError:
            log.error("[GoofFetch] cinemagoer not installed. "
                      "pip install cinemagoer")
            return self._fallback(title, year)

        try:
            ia = Cinemagoer()

            # Search for the movie
            log.info("[GoofFetch] Searching: '%s' (%d)", title, year)
            results = ia.search_movie(f"{title}")

            # Find best match by year
            movie = None
            for r in results:
                r_year = r.get("year", 0)
                r_title = r.get("title", "").lower()
                if r_year == year or (abs(r_year - year) <= 1 and
                                      title.lower() in r_title):
                    movie = r
                    break

            if movie is None and results:
                movie = results[0]  # fallback to top result
                log.warning("[GoofFetch] No exact year match, using top result: %s (%s)",
                            movie.get("title"), movie.get("year"))

            if movie is None:
                # Try direct IMDb ID lookup if we have one in the map
                known_id = _IMDB_ID_MAP.get((title.lower(), year))
                if known_id:
                    log.info("[GoofFetch] search_movie() empty — trying direct ID tt%s", known_id)
                    try:
                        movie = ia.get_movie(known_id)
                    except Exception as e:
                        log.warning("[GoofFetch] Direct ID lookup failed: %s", e)
                        movie = None

            if movie is None:
                log.warning("[GoofFetch] No results for '%s'", title)
                return self._fallback(title, year)

            # Fetch full movie data
            ia.update(movie, info=["main"])
            movie_id = movie.movieID

            # ── Title sanity check ─────────────────────────────────────
            # Guard against wrong IMDb IDs in the pool: if the fetched
            # title doesn't resemble the expected title, skip this movie
            # and fall back so we don't generate content for the wrong film.
            fetched_title = (movie.get("title") or "").lower().strip()
            expected_lower = title.lower().strip()
            if fetched_title and expected_lower:
                # Check if either contains the other, or first significant
                # word matches (handles "The X" vs "X" differences)
                exp_words = [w for w in expected_lower.split()
                             if w not in ("the", "a", "an")]
                fetch_words = [w for w in fetched_title.split()
                               if w not in ("the", "a", "an")]
                first_match = (exp_words and fetch_words
                               and exp_words[0] == fetch_words[0])
                contains = (expected_lower in fetched_title
                            or fetched_title in expected_lower)
                if not first_match and not contains:
                    log.error(
                        "[GoofFetch] TITLE MISMATCH: expected '%s' but "
                        "IMDb returned '%s' (tt%s) — skipping!",
                        title, movie.get("title"), movie_id)
                    return self._fallback(title, year)

            # ── Extract cast names (actors + character names) ──
            cast_names = self._extract_cast_names(movie)

            # Cinemagoer may return genres under "genres" or "genre" depending
            # on the fetch path; try both and normalise to a list of strings.
            raw_genres = (movie.get("genres") or movie.get("genre") or [])
            genres_list = [str(g) for g in raw_genres] if raw_genres else []
            log.info("[GoofFetch] Genres for '%s': %s", title, genres_list or "(none found)")

            movie_data = {
                "imdb_id": f"tt{movie_id}",
                "title": movie.get("title", title),
                "year": movie.get("year", year),
                "genres": genres_list,
                "plot": (movie.get("plot outline", "") or
                         (movie.get("plot", [""])[0] if movie.get("plot") else "")),
                "rating": movie.get("rating", 0),
                "kind": movie.get("kind", "movie"),
                "cast_names": cast_names,
            }

            # Fetch goofs
            log.info("[GoofFetch] Fetching goofs for '%s' (tt%s)...",
                     movie_data["title"], movie_id)
            ia.update(movie, info=["goofs"])
            raw_goofs = movie.get("goofs", {})

            goofs = []
            if isinstance(raw_goofs, dict):
                for category, goof_list in raw_goofs.items():
                    if not isinstance(goof_list, (list, tuple)):
                        goof_list = [goof_list]
                    for g in goof_list:
                        if isinstance(g, str) and g.strip():
                            goofs.append({
                                "category": _normalize_category(category),
                                "description": g.strip(),
                            })
            elif isinstance(raw_goofs, (list, tuple)):
                for g in raw_goofs:
                    if isinstance(g, str) and g.strip():
                        goofs.append({
                            "category": "Miscellaneous",
                            "description": g.strip(),
                        })

            log.info("[GoofFetch] Cinemagoer found %d raw goofs for '%s'",
                     len(goofs), movie_data["title"])

            # ── Direct HTTP fallback ──────────────────────────────────────
            # Cinemagoer's goof HTML parser breaks against IMDb's current
            # page structure. If it returned nothing, try fetching the goofs
            # page ourselves and parsing the embedded __NEXT_DATA__ JSON or
            # falling back to BeautifulSoup.
            # Also extracts genres from __NEXT_DATA__ to backfill when
            # Cinemagoer's ia.update() comes up empty (common on direct-ID path).
            if not goofs or not movie_data.get("genres"):
                if not goofs:
                    log.info("[GoofFetch] Cinemagoer returned 0 goofs — "
                             "trying direct HTTP fetch for tt%s", movie_id)
                else:
                    log.info("[GoofFetch] Genres empty — fetching via HTTP for tt%s",
                             movie_id)
                direct_goofs, direct_genres = self._fetch_goofs_direct(
                    str(movie_id), max_goofs)
                if direct_goofs:
                    goofs = direct_goofs
                    log.info("[GoofFetch] Direct fetch recovered %d goofs "
                             "(full pool)", len(goofs))
                if direct_genres and not movie_data.get("genres"):
                    movie_data["genres"] = direct_genres
                    log.info("[GoofFetch] Genres backfilled from HTTP: %s",
                             direct_genres)

            # ── Cache the FULL pool before culling ────────────────────────
            # Storing all goofs means future runs with different seeds can
            # draw fresh subsets without hitting IMDb again.
            if goofs:
                _cache_put(title, year, movie_data, goofs, cast_names)

            # ── Seed-based diversified selection ─────────────────────────
            goofs = self._diversify(goofs, max_goofs, seed=seed)
            log.info("[GoofFetch] Selected %d goofs for '%s' (seed=%s)",
                     len(goofs), movie_data["title"], seed)

            if not goofs:
                log.warning("[GoofFetch] No goofs found, using placeholder")
                goofs = self._placeholder_goofs(movie_data, max_goofs)

            return (movie_data, goofs)

        except Exception as e:
            log.error("[GoofFetch] Cinemagoer error: %s", e)
            return self._fallback(title, year)

    def _extract_cast_names(self, movie) -> list:
        """Pull top-billed actor names + character names from Cinemagoer.

        Returns a flat list of name strings for the sanitizer to strip.
        """
        names = []
        try:
            cast = movie.get("cast", [])
            for person in cast[:20]:  # top 20 billed
                # Actor name
                actor_name = str(person)
                if actor_name:
                    names.append(actor_name)
                    # Also add first/last separately for partial matches
                    parts = actor_name.split()
                    if len(parts) >= 2:
                        names.append(parts[0])   # first name
                        names.append(parts[-1])   # last name

                # Character name
                char = person.currentRole
                if char:
                    char_str = str(char)
                    if char_str and char_str.lower() not in ("", "none"):
                        names.append(char_str)
                        parts = char_str.split()
                        if len(parts) >= 2:
                            names.append(parts[0])
                            names.append(parts[-1])

            # Deduplicate, preserve order
            seen = set()
            unique = []
            for n in names:
                n_lower = n.lower().strip()
                if n_lower and len(n_lower) > 1 and n_lower not in seen:
                    seen.add(n_lower)
                    unique.append(n.strip())
            names = unique

            log.info("[GoofFetch] Extracted %d cast/character names", len(names))
        except Exception as e:
            log.warning("[GoofFetch] Cast extraction error: %s", e)

        return names

    # ──────────────────────────────────────────────────────────────────
    # Direct HTTP fallback — bypasses Cinemagoer's broken goof parser
    # ──────────────────────────────────────────────────────────────────

    def _fetch_goofs_direct(self, imdb_id: str, max_count: int):
        """Fetch IMDb goofs page directly when Cinemagoer returns nothing.

        Tries two strategies in order:
        1. Parse embedded ``__NEXT_DATA__`` JSON (fast, zero deps beyond requests)
        2. BeautifulSoup HTML parse (slower, requires bs4)

        Returns (goofs_list, genres_list).  Genres are extracted from
        __NEXT_DATA__ as a bonus when available; list is empty on failure.
        """
        try:
            import requests
        except ImportError:
            log.warning("[GoofFetch] `requests` not available — "
                        "install it with: pip install requests")
            return [], []

        # Strip any leading "tt" so we always have the bare numeric ID
        clean_id = imdb_id.lstrip("t")
        url = f"https://www.imdb.com/title/tt{clean_id}/goofs/"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

        try:
            resp = requests.get(url, headers=headers, timeout=20)
            resp.raise_for_status()
            html = resp.text
            log.info("[GoofFetch] Direct HTTP: %d bytes for tt%s",
                     len(html), clean_id)
        except Exception as e:
            log.warning("[GoofFetch] Direct HTTP request failed: %s", e)
            return [], []

        # ── Strategy 1: __NEXT_DATA__ embedded JSON ──
        nd_match = re.search(
            r'<script[^>]+id=["\']__NEXT_DATA__["\'][^>]*>\s*(\{.*?)\s*</script>',
            html, re.DOTALL
        )
        if nd_match:
            try:
                data = json.loads(nd_match.group(1))
                goofs = self._extract_goofs_next_data(data)
                genres = self._extract_genres_next_data(data)
                if goofs:
                    log.info("[GoofFetch] __NEXT_DATA__ yielded %d goofs, genres=%s",
                             len(goofs), genres or "(none)")
                    return goofs, genres
                else:
                    log.debug("[GoofFetch] __NEXT_DATA__ parsed OK but 0 goofs")
            except Exception as e:
                log.debug("[GoofFetch] __NEXT_DATA__ parse error: %s", e)

        # ── Strategy 2: BeautifulSoup HTML parse ──
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            goofs = self._parse_goofs_soup(soup)
            if goofs:
                log.info("[GoofFetch] BeautifulSoup yielded %d goofs",
                         len(goofs))
                return goofs, []
            else:
                log.debug("[GoofFetch] BeautifulSoup found 0 goofs")
        except ImportError:
            log.debug("[GoofFetch] bs4 not installed")
        except Exception as e:
            log.debug("[GoofFetch] Soup parse error: %s", e)

        log.warning("[GoofFetch] Direct fetch exhausted all strategies for tt%s",
                    clean_id)
        return [], []

    def _extract_genres_next_data(self, root: dict) -> list:
        """Extract genre strings from IMDb __NEXT_DATA__ JSON.

        IMDb embeds genres roughly at:
          props.pageProps.mainColumnData.genres.genres[].text
        We do a recursive search for any dict containing a 'genres' key
        whose value is a list of dicts with a 'text' field — that way
        we're resilient to schema changes.
        """
        found = []

        def _search(obj, depth=0):
            if depth > 20 or not obj:
                return
            if isinstance(obj, dict):
                g = obj.get("genres")
                if isinstance(g, list) and g:
                    # Could be list-of-dicts {"id": "Action", "text": "Action"}
                    # or list-of-strings ["Action", "Thriller"]
                    for item in g:
                        if isinstance(item, dict):
                            txt = item.get("text") or item.get("id") or ""
                            if txt and txt not in found:
                                found.append(str(txt))
                        elif isinstance(item, str) and item not in found:
                            found.append(item)
                    if found:
                        return   # stop once we have something
                for v in obj.values():
                    _search(v, depth + 1)
            elif isinstance(obj, list):
                for item in obj:
                    _search(item, depth + 1)

        _search(root)
        return found

    def _extract_goofs_next_data(self, root: dict) -> list:
        """Recursively mine IMDb ``__NEXT_DATA__`` for goof entries.

        IMDb's goofs page embeds data roughly as:
        props → pageProps → contentData → categories → [{id, label, section:
            {items: [{htmlContent: "..."}]}}]

        We do a recursive search so we're resilient to schema changes.
        """
        goofs = []
        seen: set = set()

        def _search(obj, cat_label="Miscellaneous", depth=0):
            if depth > 25 or not obj:
                return
            if isinstance(obj, dict):
                # When we hit a "categories" list, dive in with per-cat labels
                if "categories" in obj and isinstance(obj["categories"], list):
                    for cat_obj in obj["categories"]:
                        if isinstance(cat_obj, dict):
                            label = (
                                cat_obj.get("name")    # IMDb current field
                                or cat_obj.get("label")
                                or cat_obj.get("id")
                                or cat_label
                            )
                            _search(cat_obj, label, depth + 1)
                    return

                # When we hit an "items" list, extract goof text
                if "items" in obj and isinstance(obj["items"], list):
                    for item in obj["items"]:
                        if not isinstance(item, dict):
                            continue
                        raw = (
                            item.get("cardHtml")       # IMDb current field
                            or item.get("htmlContent")
                            or item.get("text")
                            or item.get("content")
                            or item.get("rowTitle")
                            or ""
                        )
                        if not isinstance(raw, str):
                            continue
                        # Strip HTML tags, then decode HTML entities (&#39; → ' etc.)
                        import html as _html
                        clean = _html.unescape(re.sub(r"<[^>]+>", "", raw)).strip()
                        if len(clean) > 20 and clean not in seen:
                            seen.add(clean)
                            goofs.append({
                                "category": _normalize_category(cat_label),
                                "description": clean,
                            })
                    return

                # Otherwise recurse into all values
                for v in obj.values():
                    _search(v, cat_label, depth + 1)

            elif isinstance(obj, list):
                for item in obj:
                    _search(item, cat_label, depth + 1)

        _search(root)
        return goofs

    def _parse_goofs_soup(self, soup) -> list:
        """Parse IMDb goofs page HTML with BeautifulSoup.

        Looks for:
        - ``div.ipc-html-content-inner-div``  — goof text containers
        - Section headings above each block   — category labels
        """
        goofs = []
        seen: set = set()

        content_divs = soup.find_all(
            "div",
            class_=re.compile(r"ipc-html-content")
        )
        log.debug("[GoofFetch] Soup found %d ipc-html-content divs",
                  len(content_divs))

        for div in content_divs:
            text = div.get_text(separator=" ", strip=True)
            if len(text) < 20 or text in seen:
                continue

            # Walk up the DOM to find a category heading
            cat = "Miscellaneous"
            node = div.parent
            for _ in range(10):
                if node is None:
                    break
                # Look for a sibling or parent heading element
                heading = node.find_previous_sibling(
                    ["h3", "h4", "h2"],
                )
                if heading:
                    cat_text = heading.get_text(strip=True)
                    if cat_text:
                        cat = cat_text
                    break
                # Also check for span/div with "section" in class
                section_hdr = node.find_previous_sibling(
                    attrs={"class": re.compile(r"section.?header|title.?header",
                                               re.I)}
                )
                if section_hdr:
                    cat_text = section_hdr.get_text(strip=True)
                    if cat_text:
                        cat = cat_text
                    break
                node = node.parent

            seen.add(text)
            goofs.append({
                "category": _normalize_category(cat),
                "description": text,
            })

        return goofs

    def _diversify(self, goofs, max_count, seed=None):
        """Pick goofs with maximum category diversity, randomized by seed.

        With a seed, the same seed always returns the same subset (reproducible),
        but different seeds draw different goofs from the full pool — so the
        workflow produces genuinely new content on every queue run when the user
        changes or randomizes the seed widget.
        """
        if not goofs:
            return goofs

        # Shuffle within each category bucket using the seed
        rng = random.Random(seed)
        by_cat: dict = {}
        for g in goofs:
            by_cat.setdefault(g["category"], []).append(g)
        for bucket in by_cat.values():
            rng.shuffle(bucket)

        # Rebuild a flat list: all categories interleaved so the two-pass
        # pick below still gets good diversity even when max_count is small.
        shuffled: list = []
        while any(by_cat.values()):
            for bucket in list(by_cat.values()):
                if bucket:
                    shuffled.append(bucket.pop(0))

        if len(shuffled) <= max_count:
            return shuffled

        # First pass: one per category
        seen_cats: set = set()
        diverse: list = []
        for g in shuffled:
            cat = g["category"]
            if cat not in seen_cats:
                diverse.append(g)
                seen_cats.add(cat)
                if len(diverse) >= max_count:
                    return diverse

        # Second pass: fill remaining slots
        for g in shuffled:
            if g not in diverse:
                diverse.append(g)
                if len(diverse) >= max_count:
                    return diverse

        return diverse

    def _parse_manual(self, json_str, title, year):
        """Parse manually provided JSON goofs."""
        try:
            data = json.loads(json_str.strip())
            if isinstance(data, list):
                goofs = []
                for item in data:
                    if isinstance(item, dict):
                        goofs.append({
                            "category": item.get("category", "Miscellaneous"),
                            "description": item.get("description", str(item)),
                        })
                    elif isinstance(item, str):
                        goofs.append({
                            "category": "Miscellaneous",
                            "description": item,
                        })
                movie_data = {
                    "imdb_id": "",
                    "title": title,
                    "year": year,
                    "genres": [],
                    "plot": "",
                    "rating": 0,
                    "kind": "movie",
                }
                return movie_data, goofs
        except (json.JSONDecodeError, TypeError) as e:
            log.warning("[GoofFetch] Invalid manual JSON: %s", e)

        return self._fallback(title, year)

    def _fallback(self, title, year):
        """Return empty movie data with placeholder goofs."""
        movie_data = {
            "imdb_id": "",
            "title": title,
            "year": year,
            "genres": [],
            "plot": f"A film titled {title} ({year}).",
            "rating": 0,
            "kind": "movie",
        }
        goofs = self._placeholder_goofs(movie_data, 3)
        return (movie_data, goofs)

    def _placeholder_goofs(self, movie_data, count):
        """Generate generic placeholder goofs when none are found."""
        title = movie_data["title"]
        placeholders = [
            {"category": "Continuity", "description":
             f"A drink on the table changes fill level between shots in a key scene."},
            {"category": "Revealing Mistake", "description":
             f"A crew member's shadow is briefly visible on the wall during an interior scene."},
            {"category": "Factual Error", "description":
             f"A character references a date that doesn't match the timeline established earlier."},
            {"category": "Anachronism", "description":
             f"A modern object is visible in a scene set in an earlier time period."},
            {"category": "Audio/Visual Unsync", "description":
             f"A character's lip movements don't match the dialogue in one shot."},
        ]
        return placeholders[:count]
