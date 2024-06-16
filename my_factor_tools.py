# v 1.0

import numpy as np
import tensorly as tl
from tlviz.factor_tools import degeneracy_score, factor_match_score, cosine_similarity
from tensorly.cp_tensor import CPTensor
import matcouply
from matcouply.penalties import MatricesPenalty
from copy import deepcopy
import math
from random import choice
import matplotlib.pyplot as plt
from scipy import spatial
import seaborn as sns

from matplotlib.ticker import FormatStrFormatter

######################################################################################
# Synthetic data generation
######################################################################################


class data_generator:

    def __init__(
        self,
        no_of_concepts,  # Number of structures existing in the data i.e. the rank.
        I,
        J,
        K,  # Tensor dimensions, where K denotes the no of timeslices.
        B_internal_overlap,  # Percentage that describes the overlap between the initial and final concepts in Bs.
        evolution_prob,  # probabilty of a point 'evolving' in a timeslice (only applicable in incremental drift).
        drift_indices,  # The indices that denote when concept changes in structure.
    ):

        self.I = I
        self.J = J
        self.K = K

        self.no_of_concepts = no_of_concepts

        self.B_internal_overlap = B_internal_overlap
        self.evolution_prob = evolution_prob
        self.drift_indices = drift_indices

    def _generate_As_(self):
        """
        Create and return A factor matrix.
        """

        self.A_sets = []

        for pattern_no in range(self.no_of_concepts):

            self.A_sets.append(
                np.array(
                    [
                        *range(
                            pattern_no * int(self.I / self.no_of_concepts),
                            pattern_no * int(self.I / self.no_of_concepts)
                            + int(self.I / self.no_of_concepts),
                        )
                    ]
                )
            )

        self.As = np.zeros((self.I, self.no_of_concepts))

        for pattern_no in range(self.no_of_concepts):

            for index in list(self.A_sets[pattern_no]):

                self.As[index, pattern_no] = np.random.normal(loc=0, scale=1)

    def _generate_Bs_(self):
        """
        Create and return B factor matrices.
        """

        # Generate pattern index sets

        self.B_initial_sets = []
        self.B_final_sets = []

        for pattern_no in range(self.no_of_concepts):  # For each pattern

            self.B_initial_sets.append(set())
            self.B_final_sets.append(set())

            while (
                len(self.B_initial_sets[pattern_no]) < 3
                or len(self.B_final_sets[pattern_no]) < 3
            ):  # Need to be at least 3 words in each topic

                # Generate a rondom index between the min index of this pattern and the max and divide the indices in the initial and the final set
                diff_index = int(
                    np.random.normal(
                        loc=pattern_no * int(self.J / self.no_of_concepts)
                        + (1 / 2) * int(self.J / self.no_of_concepts),
                        scale=pattern_no * int(self.J / self.no_of_concepts)
                        + (1 / 4) * int(self.J / self.no_of_concepts),
                    )
                )
                self.B_initial_sets[pattern_no] = set(
                    [*range(pattern_no * int(self.J / self.no_of_concepts), diff_index)]
                )
                self.B_final_sets[pattern_no] = set(
                    [
                        *range(
                            diff_index,
                            pattern_no * int(self.J / self.no_of_concepts)
                            + int(self.J / self.no_of_concepts),
                        )
                    ]
                )

            # Apply internal overlap (iff applicable)

            coin_result = np.random.binomial(1, 0.5)  # Toss a coin

            if coin_result == 1:

                for item in self.B_initial_sets[pattern_no]:

                    if (
                        100
                        * len(
                            self.B_initial_sets[pattern_no].intersection(
                                self.B_final_sets[pattern_no]
                            )
                        )
                        / len(self.B_initial_sets[pattern_no])
                        >= self.B_internal_overlap
                    ):
                        break

                    self.B_final_sets[pattern_no].add(item)

                for item in self.B_final_sets[pattern_no]:

                    if (
                        100
                        * len(
                            self.B_initial_sets[pattern_no].intersection(
                                self.B_final_sets[pattern_no]
                            )
                        )
                        / len(self.B_initial_sets[pattern_no])
                        >= self.B_internal_overlap
                    ):
                        break

                    self.B_initial_sets[pattern_no].add(item)

            else:

                for item in self.B_final_sets[pattern_no]:

                    if (
                        100
                        * len(
                            self.B_final_sets[pattern_no].intersection(
                                self.B_initial_sets[pattern_no]
                            )
                        )
                        / len(self.B_final_sets[pattern_no])
                        >= self.B_internal_overlap
                    ):
                        break

                    self.B_initial_sets[pattern_no].add(item)

                for item in self.B_initial_sets[pattern_no]:

                    if (
                        100
                        * len(
                            self.B_final_sets[pattern_no].intersection(
                                self.B_initial_sets[pattern_no]
                            )
                        )
                        / len(self.B_final_sets[pattern_no])
                        >= self.B_internal_overlap
                    ):
                        break

                    self.B_final_sets[pattern_no].add(item)

        # Generate lookup tables of evolution of Bs

        self.Bs_lookup = [self.B_initial_sets]
        pattern_status = ["initial" for _ in range(self.no_of_concepts)]

        self.pattern_statuses = deepcopy([pattern_status])

        items2remove = [
            set() for i in range(self.no_of_concepts)
        ]  # Careful: may work with only one incremental pattern
        items2add = [set() for i in range(self.no_of_concepts)]

        for t in range(1, self.K):

            temp = deepcopy(self.Bs_lookup[-1])

            for pattern_no in range(self.no_of_concepts):

                # Incremental
                if (
                    pattern_status[pattern_no] == "initial"
                    and t > self.drift_indices[pattern_no][0]
                ):

                    for item in temp[pattern_no]:

                        if item in items2add[pattern_no]:
                            continue

                        # change_prob = self.sigmoid_(t,self.drift_indices[pattern_no][0])

                        coin_result = np.random.binomial(1, self.evolution_prob)

                        if coin_result == 1:

                            opt_menu = [
                                0,
                                1,
                                2,
                            ]  # 0: Remove item from temp, 1: add item from final, 2: exchange

                            if (
                                self.B_final_sets[pattern_no].intersection(
                                    temp[pattern_no]
                                )
                                == self.B_final_sets[pattern_no]
                            ):
                                opt_menu.remove(1)

                            if item in self.B_final_sets[pattern_no]:
                                opt_menu.remove(0)

                            if (
                                item in self.B_final_sets[pattern_no]
                                or self.B_final_sets[pattern_no].intersection(
                                    temp[pattern_no]
                                )
                                == self.B_final_sets[pattern_no]
                            ):
                                opt_menu.remove(2)

                            if len(opt_menu) == 0:
                                continue

                            option = choice(opt_menu)

                            if option == 0:  # Remove item from temp

                                items2remove[pattern_no].add(item)

                            elif option == 1:  # Add random item from final

                                new_index = choice(
                                    list(
                                        self.B_final_sets[pattern_no].difference(
                                            temp[pattern_no]
                                        )
                                    )
                                )
                                items2add[pattern_no].add(new_index)

                            elif option == 2:  # exhcange

                                new_index = choice(
                                    list(
                                        self.B_final_sets[pattern_no].difference(
                                            temp[pattern_no]
                                        )
                                    )
                                )
                                items2remove[pattern_no].add(item)
                                items2add[pattern_no].add(new_index)

                    temp[pattern_no].difference_update(items2remove[pattern_no])
                    temp[pattern_no].update(items2add[pattern_no])

                    if self.B_final_sets[pattern_no] == temp[pattern_no]:

                        pattern_status[pattern_no] = "final"

            self.Bs_lookup.append(deepcopy(temp))
            self.pattern_statuses.append(deepcopy(pattern_status))

        # Create Bs according to the index matrices

        self.Bs = []

        factor_template = np.zeros((self.J, self.no_of_concepts))

        missing_indices = [[] for u in range(self.no_of_concepts)]
        missing_indices_timestamps = [[] for u in range(self.no_of_concepts)]

        new_indices = [[] for u in range(self.no_of_concepts)]
        new_indices_timestamps = [[] for u in range(self.no_of_concepts)]

        for t in range(self.K):

            self.Bs.append(deepcopy(factor_template))

            for pattern_no in range(self.no_of_concepts):

                if t == 0:

                    for index in self.Bs_lookup[t][pattern_no]:

                        ## Randomly chosen values - uncomment following lines

                        val = np.random.uniform(-1, 1)  # allow for negative values
                        # val = np.random.uniform(0,1)

                        self.Bs[t][index, pattern_no] = val

                else:

                    # Find missing indices

                    missing_indices[pattern_no].extend(
                        list(
                            self.Bs_lookup[t - 1][pattern_no].difference(
                                self.Bs_lookup[t][pattern_no]
                            )
                        )
                    )
                    missing_indices_timestamps[pattern_no].extend(
                        [
                            t
                            for _ in range(
                                len(
                                    self.Bs_lookup[t - 1][pattern_no].difference(
                                        self.Bs_lookup[t][pattern_no]
                                    )
                                )
                            )
                        ]
                    )

                    # find new indices

                    new_indices[pattern_no].extend(
                        list(
                            self.Bs_lookup[t][pattern_no].difference(
                                self.Bs_lookup[t - 1][pattern_no]
                            )
                        )
                    )
                    new_indices_timestamps[pattern_no].extend(
                        [
                            t
                            for _ in range(
                                len(
                                    self.Bs_lookup[t][pattern_no].difference(
                                        self.Bs_lookup[t - 1][pattern_no]
                                    )
                                )
                            )
                        ]
                    )

                    # fill out missing_indices

                    for index in missing_indices[pattern_no]:

                        if self.Bs[t - 1][index, pattern_no] < 0:
                            val2add = np.random.uniform(0, 0.2)
                        elif self.Bs[t - 1][index, pattern_no] > 0:
                            val2add = np.random.uniform(-0.2, 0)
                        else:
                            val2add = 0

                        # val2add = np.random.uniform(-0.2,0.2)
                        # while(self.Bs[t-1][index,pattern_no]+val2add < 0 or self.Bs[t-1][index,pattern_no]+val2add > 1): val2add = np.random.uniform(-0.2,0.2)

                        # self.Bs[t][index,pattern_no] = self.Bs[t-1][index,pattern_no] + val2add

                        # if self.Bs[t-1][index,pattern_no] changes sign after we add val2add, set self.Bs[t][index,pattern_no] to zero
                        if (
                            self.Bs[t - 1][index, pattern_no] + val2add < 0
                            and self.Bs[t - 1][index, pattern_no] > 0
                        ):
                            self.Bs[t][index, pattern_no] = 0
                        elif (
                            self.Bs[t - 1][index, pattern_no] + val2add > 0
                            and self.Bs[t - 1][index, pattern_no] < 0
                        ):
                            self.Bs[t][index, pattern_no] = 0
                        else:
                            self.Bs[t][index, pattern_no] = (
                                self.Bs[t - 1][index, pattern_no] + val2add
                            )

                    for index in self.Bs_lookup[t][pattern_no]:

                        if (
                            index in new_indices[pattern_no]
                            and new_indices_timestamps[pattern_no][
                                new_indices[pattern_no].index(index)
                            ]
                            == t
                        ):

                            self.Bs[t][index, pattern_no] = np.random.uniform(-0.2, 0.2)
                            # self.Bs[t][index,pattern_no] = np.random.uniform(0,0.2) # non-negative values

                        else:

                            val2add = np.random.uniform(-0.2, 0.2)

                            # while(self.Bs[t-1][index,pattern_no] + val2add < 0 ): val2add = np.random.uniform(-0.2,0.2)

                            self.Bs[t][index, pattern_no] = (
                                self.Bs[t - 1][index, pattern_no] + val2add
                            )

    def _generate_Cs_(self):
        """
        Create and return C factor matrices.
        """

        while True:

            self.Cs = []
            for _ in range(self.K):
                # Generate a 1D array of random uniform values for the diagonal
                diagonal_values = np.random.uniform(2, 15, size=self.no_of_concepts)
                # Create a diagonal matrix for these values
                diag_matrix = np.diag(diagonal_values)
                self.Cs.append(diag_matrix)

            C_matrix = self.get_C_matrix()

            max_cosine_distance = 0
            for i in range(C_matrix.shape[1]):
                for j in range(i + 1, C_matrix.shape[1]):
                    cosine_distance = np.dot(C_matrix[:, i], C_matrix[:, j]) / (
                        np.linalg.norm(C_matrix[:, i]) * np.linalg.norm(C_matrix[:, j])
                    )
                    if cosine_distance > max_cosine_distance:
                        max_cosine_distance = cosine_distance

            if max_cosine_distance < 0.8:
                break

    def generate_data(self):
        """
        Returns the generated data with given parameters.
        """

        # Generate factor matrices

        self._generate_As_()

        self._generate_Bs_()

        self._generate_Cs_()

        # Form tensor

        X = tl.zeros((self.I, self.J, self.K))

        for t in range(self.K):

            X[..., t] = self.As @ self.Cs[t] @ self.Bs[t].T

        self.data = X
        self.noiseless_data = X

        return self.data

    def plot_As(self):

        fig, axes = plt.subplots(1, 1, figsize=(5, 5))

        im = axes.imshow(self.As)

        axes.set_xticks([])
        axes.set_yticks([])

        fig.colorbar(im, ax=axes, pad=0.02)

    def plot_Bs(self):

        fig, axes = plt.subplots(1, self.no_of_concepts)
        plt.tight_layout()

        for i in range(self.no_of_concepts):

            B_2_plot_gnd_truth = form_plotting_B(self.Bs, i, self.J, self.K)
            im = sns.heatmap(
                B_2_plot_gnd_truth.T,
                ax=axes[i],
                cbar=False,
                cmap="viridis",
                rasterized=True,
            )
            axes[i].tick_params(left=False, bottom=True)
            axes[i].patch.set_edgecolor("black")
            axes[i].set_yticks([])
            axes[i].patch.set_linewidth(1.5)
            axes[i].set_xticks([4, 9, 14, 19], labels=[5, 10, 15, 20], fontsize=3.5)
            axes[i].set_xlabel(r"time", fontsize=6)
            axes[i].set_ylabel(r"words", fontsize=6)

    def plot_Cs(self):

        fig, axes = plt.subplots(1, 1)

        C_tensor = self.get_C_matrix()

        axes.set_title(label="Columns of $C$", pad=3.5, fontsize=6)
        axes.set_xticks([*range(1, self.K + 1)])
        axes.set_xlabel("time", fontsize=6)
        axes.set_ylabel("strength", fontsize=6)

        for pattern_no in range(self.no_of_concepts):

            axes.plot(
                [*range(self.K)],
                C_tensor[:, pattern_no],
                label=f"$c_{pattern_no}$",
                linewidth=0.5,
                linestyle="dotted",
            )

        axes.set_xticks([])
        axes.set_xticklabels([])
        axes.set_xticks([*range(0, self.K, 2)])
        axes.set_yticks([0, 10, 20])
        axes.set_xticklabels([*range(1, self.K + 1, 2)], fontsize=3.5)

        axes.legend(fontsize=4.5)
        plt.show()

    def get_C_matrix(self):

        new_Cs = tl.zeros((self.K, self.no_of_concepts))

        for t in range(self.K):

            new_Cs[t, :] = np.diag(self.Cs[t])

        return new_Cs


######################################################################################
# Metrics
######################################################################################


def check_degenerate(factors, threshold=-0.85):
    """
    Check solution for degenerecy (just a wrapper for tlviz degeneracy score).
    """

    A = factors[2]
    B = factors[1]
    D = factors[0]

    new_B = np.vstack(B)
    decomp = CPTensor((np.ones(A.shape[1]), (D, new_B, A)))

    if degeneracy_score(decomp) < threshold:
        return True
    else:
        return False


def cosine_sim(factor1, factor2):
    """
    Receive two factors and compute the cosine similarity of them using
    """

    from scipy.spatial import distance

    factor1_flat = deepcopy(factor1.flatten())

    factor2_flat = deepcopy(factor2.flatten())

    return 1 - distance.cosine(factor1_flat, factor2_flat)


def get_all_fms(gnd_factors, est_factors, skip_mode=None):

    (A, B_is, C) = gnd_factors

    # Normalize factors before forming CP tensors
    B_i_norms = [tl.norm(B_i, axis=0) for B_i in B_is]
    B_is = [B_i / B_i_norm for B_i, B_i_norm in zip(B_is, B_i_norms)]
    B_i_norms = tl.stack(B_i_norms)

    A_norm = tl.norm(A, axis=0)
    A = A / A_norm

    C_norm = tl.norm(C, axis=0)
    C = C / C_norm

    (A2, B_is2, C2) = est_factors

    B_i_norms2 = [tl.norm(B_i2, axis=0) for B_i2 in B_is2]
    B_is2 = [B_i2 / B_i_norm2 for B_i2, B_i_norm2 in zip(B_is2, B_i_norms2)]
    B_i_norms2 = tl.stack(B_i_norms2)

    A_norm2 = tl.norm(A2, axis=0)
    A2 = A2 / A_norm2

    C_norm2 = tl.norm(C2, axis=0)
    C2 = C2 / C_norm2

    cp_tensor1 = (
        (np.array([1.0] * A.shape[1])),
        (A, np.vstack(np.array(deepcopy(B_is))), C),
    )
    cp_tensor2 = (
        (np.array([1.0] * A.shape[1])),
        (A2, np.vstack(np.array(deepcopy(B_is2))), C2),
    )

    return {
        "full": factor_match_score(
            cp_tensor1,
            cp_tensor2,
            absolute_value=True,
            consider_weights=False,
            skip_mode=skip_mode,
        ),
        "C": cosine_similarity(A, A2, absolute_value=True),
        "B": cosine_similarity(
            np.vstack(np.array(deepcopy(B_is))),
            np.vstack(np.array(deepcopy(B_is2))),
            absolute_value=True,
        ),
        "A": cosine_similarity(C, C2, absolute_value=True),
    }


######################################################################################
# AO-ADMM penalties
######################################################################################


class myTemporalSmoothnessPenalty(MatricesPenalty):
    def __init__(
        self, smoothness_l, aux_init="random_uniform", dual_init="random_uniform"
    ):
        super().__init__(aux_init=aux_init, dual_init=dual_init)
        self.smoothness_l = smoothness_l

    @copy_ancestor_docstring
    def factor_matrices_update(self, factor_matrices, feasibility_penalties, auxes):

        # factor_matrices: factor + mus
        # feasability_penalties: rhos
        # auxes: -||-

        # rhs = [rhos[i] * factor_matrices[i] for i in range(len(B_is))]

        B_is = factor_matrices
        rhos = feasibility_penalties

        rhs = [rhos[i] * factor_matrices[i] for i in range(len(B_is))]

        # Construct matrix A to peform gaussian elimination on

        A = np.zeros((len(B_is), len(B_is)))

        for i in range(len(B_is)):
            for j in range(len(B_is)):
                if i == j:
                    A[i, j] = 4 * self.smoothness_l + rhos[i]
                elif i == j - 1 or i == j + 1:
                    A[i, j] = -2 * self.smoothness_l
                else:
                    pass

        A[0, 0] -= 2 * self.smoothness_l
        A[len(B_is) - 1, len(B_is) - 1] -= 2 * self.smoothness_l

        # Peform GE

        for k in range(1, A.shape[-1]):
            m = A[k, k - 1] / A[k - 1, k - 1]

            A[k, :] = A[k, :] - m * A[k - 1, :]
            rhs[k] = rhs[k] - m * rhs[k - 1]  # Also update the respective rhs!

        # Back-substitution

        new_ZBks = [np.empty_like(B_is[i]) for i in range(len(B_is))]

        new_ZBks[-1] = rhs[-1] / A[-1, -1]
        q = new_ZBks[-1]

        for i in range(A.shape[-1] - 2, -1, -1):
            q = (rhs[i] - A[i, i + 1] * q) / A[i, i]
            new_ZBks[i] = q

        return new_ZBks

    def penalty(self, x):
        penalty = 0
        for x1, x2 in zip(x[:-1], x[1:]):
            penalty += np.sum((x1 - x2) ** 2)
        return self.smoothness_l * penalty


######################################################################################
# Plotting
######################################################################################
def form_plotting_B(B_list, pattern_no, J, K):
    """
    Takes as input a list of B factors and return a matrix containing
    the pattern_no-th column of each factor matrix.
    """

    matrix2return = np.zeros((K, J))

    for k in range(K):

        matrix2return[k, :] = B_list[k][:, pattern_no].T

    return matrix2return


def plot_convergence(diagnostics_per_init, factors_per_init, zoom_in_to_first_n=50):
    """
    Plot convergence of all initializations in the following format:

    rel_sse | parafac2 constraint feasiblity gap
    -----------------------------------------------
    total_loss | temporal smoothness feasibility gap

    Degenerate cases are not plotted.
    """

    inits2ignore = [False] * len(factors_per_init)

    for init_no in range(len(factors_per_init)):

        if check_degenerate(factors_per_init[init_no]) == True:

            inits2ignore[init_no] = True
            print(f"Initialization {init_no} is degenerate and will not be plotted.")

    fig, axs = plt.subplot_mosaic(
        [["rec_errors", "parafac2_feasibility"], ["reg_loss", "smoothness_feasiblity"]],
        figsize=(12, 6),
    )

    plt.tight_layout()

    max_iters = max([diag.n_iter for diag in diagnostics_per_init])

    all_min_cur_error = []
    all_min_cur_parafac2_feasibility = []
    all_min_cur_reg_loss = []
    all_min_cur_smoothness_feasibility = []

    all_max_cur_error = []
    all_max_cur_parafac2_feasibility = []
    all_max_cur_reg_loss = []
    all_max_cur_smoothness_feasibility = []

    all_median_cur_error = []
    all_median_cur_parafac2_feasibility = []
    all_median_cur_reg_loss = []
    all_median_cur_smoothness_feasibility = []

    for iter_no in range(max_iters):

        # Form a list of all diagnostics for this initialization at the current iteration

        cur_rer_errors = []
        cur_parafac2_feasibility = []
        cur_reg_loss = []
        cur_smoothness_feasibility = []

        for init_no in range(len(factors_per_init)):

            if (
                inits2ignore[init_no] == False
                and iter_no <= diagnostics_per_init[init_no].n_iter
            ):

                cur_rer_errors.append(diagnostics_per_init[init_no].rec_errors[iter_no])
                # cur_rer_errors.append(diagnostics_per_init[init_no].un_rec_errors[iter_no])
                cur_parafac2_feasibility.append(
                    diagnostics_per_init[init_no].feasibility_gaps[iter_no][1][0]
                )
                try:
                    cur_smoothness_feasibility.append(
                        diagnostics_per_init[init_no].feasibility_gaps[iter_no][1][1]
                    )
                except:
                    pass
                cur_reg_loss.append(
                    diagnostics_per_init[init_no].regularized_loss[iter_no]
                )

        all_min_cur_error.append(min(cur_rer_errors))
        all_max_cur_error.append(max(cur_rer_errors))
        all_median_cur_error.append(np.median(cur_rer_errors))

        all_min_cur_parafac2_feasibility.append(min(cur_parafac2_feasibility))
        all_max_cur_parafac2_feasibility.append(max(cur_parafac2_feasibility))
        all_median_cur_parafac2_feasibility.append(np.median(cur_parafac2_feasibility))

        all_min_cur_reg_loss.append(min(cur_reg_loss))
        all_max_cur_reg_loss.append(max(cur_reg_loss))
        all_median_cur_reg_loss.append(np.median(cur_reg_loss))

        try:
            all_min_cur_smoothness_feasibility.append(min(cur_smoothness_feasibility))
            all_max_cur_smoothness_feasibility.append(max(cur_smoothness_feasibility))
            all_median_cur_smoothness_feasibility.append(
                np.median(cur_smoothness_feasibility)
            )
        except:
            pass

    # Plot the area between min and max errors and the median error
    axs["rec_errors"].fill_between(
        range(max_iters),
        all_min_cur_error,
        all_max_cur_error,
        color="tab:blue",
        alpha=0.2,
    )
    axs["rec_errors"].plot(
        range(max_iters), all_median_cur_error, color="tab:blue", label="rec_errors"
    )
    axs["rec_errors"].set_title("Fidelity term")

    # Plot the area between min and max parafac2 feasibility and the median parafac2 feasibility
    try:
        axs["parafac2_feasibility"].fill_between(
            range(max_iters),
            all_min_cur_parafac2_feasibility,
            all_max_cur_parafac2_feasibility,
            color="tab:orange",
            alpha=0.2,
        )
        axs["parafac2_feasibility"].plot(
            range(max_iters),
            all_median_cur_parafac2_feasibility,
            color="tab:orange",
            label="parafac2_feasibility",
        )
        axs["parafac2_feasibility"].set_title("Parafac2 feasibility")
    except:
        pass

    # Plot the area between min and max reg loss and the median reg loss
    axs["reg_loss"].fill_between(
        range(max_iters),
        all_min_cur_reg_loss,
        all_max_cur_reg_loss,
        color="tab:green",
        alpha=0.2,
    )
    axs["reg_loss"].plot(
        range(max_iters), all_median_cur_reg_loss, color="tab:green", label="reg_loss"
    )
    axs["reg_loss"].set_title("Total loss")

    # Plot the area between min and max smoothness feasibility and the median smoothness feasibility
    try:
        axs["smoothness_feasiblity"].fill_between(
            range(max_iters),
            all_min_cur_smoothness_feasibility,
            all_max_cur_smoothness_feasibility,
            color="tab:red",
            alpha=0.2,
        )
        axs["smoothness_feasiblity"].plot(
            range(max_iters),
            all_median_cur_smoothness_feasibility,
            color="tab:red",
            label="smoothness_feasiblity",
        )
        axs["smoothness_feasiblity"].set_title("Smoothness feasibility")
    except:
        pass

    zoomed_in_rec_error = fig.add_axes([0.285, 0.765, 0.2, 0.2])

    zoomed_in_rec_error.fill_between(
        range(max_iters)[:zoom_in_to_first_n],
        all_min_cur_error[:zoom_in_to_first_n],
        all_max_cur_error[:zoom_in_to_first_n],
        color="tab:blue",
        alpha=0.2,
    )
    zoomed_in_rec_error.plot(
        range(max_iters)[:zoom_in_to_first_n],
        all_median_cur_error[:zoom_in_to_first_n],
        color="tab:blue",
        label="rec_errors",
    )
    # zoomed_in_rec_error.set_yticks([0.2, 0.4, 0.6, 0.8])

    try:
        zoomed_in_parafac2_feasibility = fig.add_axes([0.778, 0.765, 0.2, 0.2])
        zoomed_in_parafac2_feasibility.fill_between(
            range(max_iters)[:zoom_in_to_first_n],
            all_min_cur_parafac2_feasibility[:zoom_in_to_first_n],
            all_max_cur_parafac2_feasibility[:zoom_in_to_first_n],
            color="tab:orange",
            alpha=0.2,
        )
        zoomed_in_parafac2_feasibility.plot(
            range(max_iters)[:zoom_in_to_first_n],
            all_median_cur_parafac2_feasibility[:zoom_in_to_first_n],
            color="tab:orange",
            label="parafac2_feasibility",
        )
        zoomed_in_parafac2_feasibility.set_yticks([0.2, 0.4, 0.6, 0.8])
        zoomed_in_parafac2_feasibility.set_yscale("log")
    except:
        pass

    zoomed_in_reg_loss = fig.add_axes([0.285, 0.278, 0.2, 0.2])
    zoomed_in_reg_loss.fill_between(
        range(max_iters)[:zoom_in_to_first_n],
        all_min_cur_reg_loss[:zoom_in_to_first_n],
        all_max_cur_reg_loss[:zoom_in_to_first_n],
        color="tab:green",
        alpha=0.2,
    )
    zoomed_in_reg_loss.plot(
        range(max_iters)[:zoom_in_to_first_n],
        all_median_cur_reg_loss[:zoom_in_to_first_n],
        color="tab:green",
        label="reg_loss",
    )
    zoomed_in_reg_loss.set_yscale("log")

    try:
        zoomed_in_smoothness_feasiblity = fig.add_axes([0.778, 0.278, 0.2, 0.2])
        zoomed_in_smoothness_feasiblity.fill_between(
            range(max_iters)[:zoom_in_to_first_n],
            all_min_cur_smoothness_feasibility[:zoom_in_to_first_n],
            all_max_cur_smoothness_feasibility[:zoom_in_to_first_n],
            color="tab:red",
            alpha=0.2,
        )
        zoomed_in_smoothness_feasiblity.plot(
            range(max_iters)[:zoom_in_to_first_n],
            all_median_cur_smoothness_feasibility[:zoom_in_to_first_n],
            color="tab:red",
            label="smoothness_feasiblity",
        )
        zoomed_in_smoothness_feasiblity.set_yscale("log")
    except:
        pass

    plt.show()


# Plot factors


def plot_factors(factors):

    import matplotlib.gridspec as gridspec

    # Normalize factors
    A = deepcopy(factors[2])
    B = deepcopy(factors[1])
    C = deepcopy(factors[0])

    A = A / np.linalg.norm(A, axis=0)
    C = C / np.linalg.norm(C, axis=0)
    for k in range(len(B)):
        B[k] = B[k] / np.linalg.norm(B[k], axis=0)

    # Create a GridSpec with 3 rows and 2 columns
    gs = gridspec.GridSpec(A.shape[1] + 1, 2)

    # Create the subplots
    fig = plt.figure(figsize=(20, 8))
    for i in range(A.shape[1]):
        ax_B1 = fig.add_subplot(gs[i, :])
        B1 = form_plotting_B(B, i, B[0].shape[0], len(B))
        ax_B1.imshow(B1, cmap="viridis")

    ax_A = fig.add_subplot(gs[-1, 0])  # A is in the first column of the third row
    ax_C = fig.add_subplot(gs[-1, 1])  # C is in the second column of the third row

    # Adjust the spacing between the plots
    plt.subplots_adjust(hspace=-0.1, wspace=0.2)  # Adjust the spacing here

    # Prep A
    # barplot of each columns of A
    for i in range(A.shape[1]):
        ax_A.bar(np.arange(A.shape[0]), A[:, i])

    # Prep C
    # lineplot of each columns of C
    for i in range(A.shape[1]):
        ax_C.plot(np.arange(C.shape[0]), C[:, i])


######################################################################################
# Missing data
######################################################################################


def get_random_mask(data, perc):

    while True:

        # Calculate the total number of elements and the number of NaNs
        total_elements = np.prod(data.shape)
        num_NaNs = int(total_elements * perc / 100)

        # create a flat array with the desired number of NaNs and ones
        mask_flat = np.array([np.NaN] * num_NaNs + [1] * (total_elements - num_NaNs))

        # Shuffle the array to distribute the zeros randomly
        np.random.shuffle(mask_flat)

        # Reshape the array to match the shape of the data
        my_missing_mask = mask_flat.reshape(data.shape)

        # check if any of the fibers is completely missing

        mode_0_missing = False
        for k in range(my_missing_mask.shape[2]):
            for j in range(my_missing_mask.shape[1]):
                if np.nansum(my_missing_mask[:, j, k]) == 0:
                    mode_0_missing = True
                    break
            if mode_0_missing:
                break

        mode_1_missing = False
        for k in range(my_missing_mask.shape[2]):
            for i in range(my_missing_mask.shape[0]):
                if np.nansum(my_missing_mask[i, :, k]) == 0:
                    mode_1_missing = True
                    break
            if mode_1_missing:
                break

        mode_2_missing = False
        for j in range(my_missing_mask.shape[1]):
            for i in range(my_missing_mask.shape[0]):
                if np.nansum(my_missing_mask[i, j, :]) == 0:
                    mode_2_missing = True
                    break
            if mode_2_missing:
                break

        if not mode_0_missing and not mode_1_missing and not mode_2_missing:
            break

    return my_missing_mask


def get_mixed_mask(data, perc):

    while True:

        # how may entries are perc % of the total number of entries
        n_entries = int((perc / 100) * data.size)
        half_n_entries = int(0.5 * n_entries)

        n_of_fibers = int(half_n_entries / data.shape[1])

        mask = np.ones_like(data)

        # randomly choose n_of_fibers tuples (i,j), where i is in range(data.shape[0]) and j is in range(data.shape[2])
        fibers = set()
        while len(fibers) < n_of_fibers:
            i = np.random.randint(data.shape[0])
            k = np.random.randint(data.shape[2])
            if (i, k) not in fibers:
                fibers.add((i, k))

        for i, k in fibers:
            mask[i, :, k] = 0

        # print(f'rem: {n_entries - np.count_nonzero(mask == 0)}')
        while np.count_nonzero(mask == 0) < n_entries:
            # print(f'{np.count_nonzero(mask == 0)}/{n_entries}')
            i = np.random.randint(data.shape[0])
            j = np.random.randint(data.shape[1])
            k = np.random.randint(data.shape[2])
            if mask[i, j, k] == 0:
                continue
            mask[i, j, k] = 0

        mode_0_missing = False
        # mode 0
        for k in range(mask.shape[-1]):
            if mask[:, :, k].sum() == 0:
                mode_0_missing = True
            break

        mode_2_missing = False
        # mode 2
        for i in range(mask.shape[0]):
            if mask[i, :, :].sum() == 0:
                mode_2_missing = True
                break

        mode_1_missing = False
        # mode 2
        for j in range(mask.shape[1]):
            if mask[:, j, :].sum() == 0:
                mode_1_missing = True
                break

        if mode_0_missing or mode_2_missing or mode_1_missing:
            continue

        mode_0_missing = False
        for k in range(mask.shape[2]):
            for j in range(mask.shape[1]):
                if mask[:, j, k].sum() == 0:
                    break
            if mode_0_missing:
                break

        if mode_0_missing:
            continue

        break

    mask[mask == 0] = np.NaN
    mask[mask == 1] = 1
    return mask


def get_fiber_mask(data, mode, perc):

    while True:

        mask = np.ones_like(data)

        if mode == 0:

            _ = np.random.choice(
                [0, 1], size=mask[:, :, 0].shape, p=[perc / 100, 1 - perc / 100]
            )

            for k in range(mask.shape[-1]):
                mask[:, :, k] = _

        elif mode == 1:

            _ = np.random.choice(
                [0, 1], size=mask[:, 0, :].shape, p=[perc / 100, 1 - perc / 100]
            )

            for j in range(mask.shape[1]):
                mask[:, j, :] = _

        elif mode == 2:

            _ = np.random.choice(
                [0, 1], size=mask[0, :, :].shape, p=[perc / 100, 1 - perc / 100]
            )

            for i in range(mask.shape[0]):
                mask[i, :, :] = _

        # replace all zeros with NaNs

        mask[mask == 0] = np.NaN

        # check if any of the slices is fully missing

        mode_0_missing = False
        # mode 0
        for k in range(mask.shape[-1]):
            if mask[:, :, k].sum() == 0:
                mode_0_missing = True
            break

        mode_1_missing = False
        # mode 1
        for j in range(mask.shape[1]):
            if mask[:, j, :].sum() == 0:
                mode_1_missing = True
                break

        mode_2_missing = False
        # mode 2
        for i in range(mask.shape[0]):
            if mask[i, :, :].sum() == 0:
                mode_2_missing = True
                break

        if mode_0_missing or mode_1_missing or mode_2_missing:
            continue

        break

    return mask
