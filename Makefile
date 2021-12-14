## Modifiable
##==============================================================================

# Nom du programme
PROG=tp
SHELL:=/bin/bash
# Dossiers
SRCDIR 	= src/
BINDIR 	= bin/
SAVEDIR = save/

# Utilisateur
FISRTNAME = Arthur
LASTNAME = Viens

# Options
OPT = -Ieigen-3.3.7/ -W -Wall -ansi -pedantic -g -v -fdiagnostics-color -std=c++11 -O2

# Compress extension
COMP = .tar.gz

## Section stable
##==============================================================================

CC = mpicxx
RM = rm -f
CP = cp -f
TAR = tar -zcvf

SRC = $(wildcard $(SRCDIR)*.cpp)
HEAD = $(wildcard $(SRCDIR)*.h)
OBJ = $(subst $(SRCDIR), $(BINDIR), $(SRC:.cpp=.o))

DIRS = $(SRCDIR) $(BINDIR) $(SAVEDIR)

.PHONY : all clean save restore give help show

all : $(PROG)

# Règle pour générer les dossiers
#--------------------------------------------------------
$(DIRS) :
	mkdir -p $@

# Règle pour générer les fichiers objets (.o).
#--------------------------------------------------------
obj : $(OBJ) | $(BINDIR)

$(BINDIR)%.o : $(SRCDIR)%.cpp $(HEAD) | $(BINDIR)
	$(CC) $(OPT) -c $< -o $@

# Règle pour générer l'exécutable
#--------------------------------------------------------
$(PROG) : $(OBJ)
	@module load gcc9.2
	@module load openmpi3.0.1-gcc7.3.0
	$(CC) $^ -o $@.out
	@echo
	@echo Entrez ./$@.out pour exécuter le programme.

# Règle pour effacer les fichiers temporaires
#--------------------------------------------------------
clean :
	find . -type f \( -name '*.o' -o -name '*\*\~' -o -name '*.bak' -o -name '*.old' -o -regex "^\#.*\#$$" \) -delete
	$(RM) $(PROG)

# Règle pour générer une sauvegarde
#--------------------------------------------------------
save : | $(SAVEDIR)
	$(CP) $(SRCDIR)* $(SAVEDIR)

# Règle pour restaurer une sauvegarde
#--------------------------------------------------------
restore : | $(SAVEDIR)
	$(CP) $(SAVEDIR)* $(SRCDIR)

# Règle pour générer une archive
#--------------------------------------------------------
give : readme
	$(TAR) $(LASTNAME)$(FISRTNAME)-$(PROG)$(COMP) $(SRCDIR) Makefile README.md

# Règle pour générer le README
#--------------------------------------------------------
readme :
	touch README.md

# Règle pour afficher l'aide
#--------------------------------------------------------
help :
	@echo 'Makefile pour des programmes C version 0.1'
	@echo 'Arthur Viens <abdulhamid@eisti.eu>'
	@echo
	@echo 'Utilisation: make [CIBLE]'
	@echo 'CIBLES:'
	@echo '  all       (=make) compile et édite les liens'
	@echo '  objs      Compile seulement (aucune édition des liens).'
	@echo "  clean     Nettoye les fichiers objets, temporaires et l'exécutable."
	@echo '  show      Affiche les variables.'
	@echo '  help      Affiche ce message.'
	@echo '  save      Crée une sauvegarde de src/ dans save/.'
	@echo '  restore   Restaure les fichiers du dossier save/.'
	@echo '  give      Génère une archive du projet/programme.'
	@echo '  readme    Génère un fichier README.md.'

# Règle pour afficher les variables
#--------------------------------------------------------
show :
	@echo 'PROG        :' $(PROG)
	@echo 'SRCDIR      :' $(SRCDIR)
	@echo 'BINDIR      :' $(BINDIR)
	@echo 'SAVEDIR     :' $(SAVEDIR)
	@echo 'HEAD        :' $(HEAD)
	@echo 'SRC         :' $(SRC)
	@echo 'OBJ         :' $(OBJ)
	@echo 'CC          :' $(CC)
	@echo 'OPT         :' $(OPT)
