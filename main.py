import numpy as np
import matplotlib.pyplot as plt
import random
import time
from mpl_toolkits.mplot3d import Axes3D


# ABC Algoritması sınıfı
class ArtificialBeeColony:
    def __init__(self, objective_func, bounds, colony_size=6, max_iter=100, limit=None):
        """
        ABC Algoritması Başlatma

        Parameters:
        - objective_func: Optimize edilecek fonksiyon
        - bounds: Değişkenlerin alt ve üst sınırları [(min1, max1), (min2, max2), ...]
        - colony_size: Koloni büyüklüğü (toplam arı sayısı)
        - max_iter: Maksimum iterasyon sayısı
        - limit: Kaynak geliştirilememe limiti
        """
        self.objective_func = objective_func
        self.bounds = bounds
        self.colony_size = colony_size
        self.max_iter = max_iter
        self.dimension = len(bounds)

        # Koloni yapısı: yarısı görevli, yarısı gözcü arı
        self.employed_bees = colony_size // 2
        self.onlooker_bees = colony_size // 2

        # Limit değeri belirleme: (Koloni Sayısı × Boyut) / 2
        if limit is None:
            self.limit = (self.colony_size * self.dimension) // 2
        else:
            self.limit = limit

        # Başlangıç değerleri
        self.food_sources = None
        self.fitness_values = None
        self.trial_counters = None
        self.global_best_solution = None
        self.global_best_fitness = float('inf')
        self.convergence_curve = []

    def initialize_food_sources(self):
        """
        Başlangıç yiyecek kaynaklarını oluşturma

        Formül: x_ij = x_j_min + rand(0,1) * (x_j_max - x_j_min)
        """
        self.food_sources = np.zeros((self.employed_bees, self.dimension))
        for i in range(self.employed_bees):
            for j in range(self.dimension):
                min_val, max_val = self.bounds[j]
                self.food_sources[i, j] = min_val + random.random() * (max_val - min_val)

        # Fitness değerlerini hesapla
        self.calculate_fitness()

        # Trial sayaçlarını sıfırla
        self.trial_counters = np.zeros(self.employed_bees)

    def calculate_objective(self, solution):
        """
        Çözümün amaç fonksiyonu değerini hesaplama
        """
        return self.objective_func(solution)

    def calculate_fitness(self):
        """
        Yiyecek kaynaklarının fitness değerlerini hesaplama

        Fitness formülü:
        fitness_i = 1/(1+f_i) if f_i >= 0
        fitness_i = 1 + abs(f_i) if f_i < 0
        """
        self.fitness_values = np.zeros(self.employed_bees)

        for i in range(self.employed_bees):
            f_i = self.calculate_objective(self.food_sources[i])

            # Fitness değerini hesapla
            if f_i >= 0:
                self.fitness_values[i] = 1 / (1 + f_i)
            else:
                self.fitness_values[i] = 1 + abs(f_i)

            # Global en iyi çözümü güncelle
            if f_i < self.global_best_fitness:
                self.global_best_fitness = f_i
                self.global_best_solution = self.food_sources[i].copy()

    def employed_bee_phase(self):
        """
        Görevli arı evresi: Lokal arama yapılır
        """
        for i in range(self.employed_bees):
            # Rastgele bir boyut seç
            j = random.randint(0, self.dimension - 1)

            # Rastgele bir komşu kaynak seç (i'den farklı)
            k = i
            while k == i:
                k = random.randint(0, self.employed_bees - 1)

            # Yeni kaynak adayını oluştur
            # Formül: v_ij = x_ij + φ_ij * (x_ij - x_kj)
            phi_ij = random.uniform(-1, 1)
            candidate = self.food_sources[i].copy()
            candidate[j] = self.food_sources[i, j] + phi_ij * (self.food_sources[i, j] - self.food_sources[k, j])

            # Sınır kontrolü
            min_val, max_val = self.bounds[j]
            if candidate[j] < min_val:
                candidate[j] = min_val
            elif candidate[j] > max_val:
                candidate[j] = max_val

            # Yeni çözümün fitness değerini hesapla
            f_candidate = self.calculate_objective(candidate)

            if f_candidate >= 0:
                fitness_candidate = 1 / (1 + f_candidate)
            else:
                fitness_candidate = 1 + abs(f_candidate)

            # Açgözlü seçim: Eğer yeni çözüm daha iyiyse kabul et
            if fitness_candidate > self.fitness_values[i]:
                self.food_sources[i] = candidate
                self.fitness_values[i] = fitness_candidate
                self.trial_counters[i] = 0  # Trial sayacını sıfırla

                # Global en iyi çözümü güncelle
                if f_candidate < self.global_best_fitness:
                    self.global_best_fitness = f_candidate
                    self.global_best_solution = candidate.copy()
            else:
                self.trial_counters[i] += 1  # Trial sayacını artır

    def onlooker_bee_phase(self):
        """
        Gözcü arı evresi: Rulet tekerleği ile kaynak seçimi
        """
        # Fitness değerlerine göre seçim olasılıklarını hesapla
        total_fitness = np.sum(self.fitness_values)

        # Sıfır bölme hatasını önle
        if total_fitness == 0:
            probabilities = np.ones(self.employed_bees) / self.employed_bees
        else:
            probabilities = self.fitness_values / total_fitness

        # Kümülatif olasılıkları hesapla
        cumulative_probabilities = np.cumsum(probabilities)

        for _ in range(self.onlooker_bees):
            # Rulet tekerleği seçimi
            rand_val = random.random()
            selected_idx = 0
            for i in range(len(cumulative_probabilities)):
                if rand_val <= cumulative_probabilities[i]:
                    selected_idx = i
                    break

            # Seçilen kaynak için lokal arama yap
            j = random.randint(0, self.dimension - 1)

            # Rastgele bir komşu kaynak seç (selected_idx'ten farklı)
            k = selected_idx
            while k == selected_idx:
                k = random.randint(0, self.employed_bees - 1)

            # Yeni kaynak adayını oluştur
            phi_ij = random.uniform(-1, 1)
            candidate = self.food_sources[selected_idx].copy()
            candidate[j] = self.food_sources[selected_idx, j] + phi_ij * (
                        self.food_sources[selected_idx, j] - self.food_sources[k, j])

            # Sınır kontrolü
            min_val, max_val = self.bounds[j]
            if candidate[j] < min_val:
                candidate[j] = min_val
            elif candidate[j] > max_val:
                candidate[j] = max_val

            # Yeni çözümün fitness değerini hesapla
            f_candidate = self.calculate_objective(candidate)

            if f_candidate >= 0:
                fitness_candidate = 1 / (1 + f_candidate)
            else:
                fitness_candidate = 1 + abs(f_candidate)

            # Açgözlü seçim: Eğer yeni çözüm daha iyiyse kabul et
            if fitness_candidate > self.fitness_values[selected_idx]:
                self.food_sources[selected_idx] = candidate
                self.fitness_values[selected_idx] = fitness_candidate
                self.trial_counters[selected_idx] = 0  # Trial sayacını sıfırla

                # Global en iyi çözümü güncelle
                if f_candidate < self.global_best_fitness:
                    self.global_best_fitness = f_candidate
                    self.global_best_solution = candidate.copy()
            else:
                self.trial_counters[selected_idx] += 1  # Trial sayacını artır

    def scout_bee_phase(self):
        """
        Kaşif arı evresi: Limit değerini aşan kaynaklar için yeni kaynak oluşturma
        """
        for i in range(self.employed_bees):
            if self.trial_counters[i] >= self.limit:
                # Yeni rastgele bir kaynak oluştur
                for j in range(self.dimension):
                    min_val, max_val = self.bounds[j]
                    self.food_sources[i, j] = min_val + random.random() * (max_val - min_val)

                # Yeni fitness değerini hesapla
                f_i = self.calculate_objective(self.food_sources[i])

                if f_i >= 0:
                    self.fitness_values[i] = 1 / (1 + f_i)
                else:
                    self.fitness_values[i] = 1 + abs(f_i)

                # Trial sayacını sıfırla
                self.trial_counters[i] = 0

                # Global en iyi çözümü güncelle
                if f_i < self.global_best_fitness:
                    self.global_best_fitness = f_i
                    self.global_best_solution = self.food_sources[i].copy()

    def optimize(self):
        """
        ABC Algoritması ile optimizasyonu çalıştırma
        """
        # Başlangıç zamanı
        start_time = time.time()

        # Başlangıç yiyecek kaynaklarını oluştur
        self.initialize_food_sources()

        print("ABC Algoritması Başlatıldı")
        print(f"Koloni Boyutu: {self.colony_size}")
        print(f"Görevli Arı Sayısı: {self.employed_bees}")
        print(f"Gözcü Arı Sayısı: {self.onlooker_bees}")
        print(f"Limit Değeri: {self.limit}")
        print(f"Maksimum İterasyon: {self.max_iter}")
        print("=" * 50)

        # İterasyonları çalıştır
        for iter in range(self.max_iter):
            # Görevli arı evresi
            self.employed_bee_phase()

            # Gözcü arı evresi
            self.onlooker_bee_phase()

            # Kaşif arı evresi
            self.scout_bee_phase()

            # Yakınsama eğrisi için global en iyi değeri kaydet
            self.convergence_curve.append(self.global_best_fitness)

            # Her 10 iterasyonda bir ilerlemeyi yazdır
            if (iter + 1) % 10 == 0:
                print(f"Iterasyon {iter + 1}/{self.max_iter}, En İyi Değer: {self.global_best_fitness:.6f}")

        # Çalışma süresini hesapla
        self.execution_time = time.time() - start_time

        print("=" * 50)
        print("Optimizasyon Tamamlandı!")
        print(f"Toplam Çalışma Süresi: {self.execution_time:.4f} saniye")
        print(f"En İyi Çözüm: {self.global_best_solution}")
        print(f"En İyi Değer: {self.global_best_fitness:.6f}")

        return self.global_best_solution, self.global_best_fitness


# Düzeltilmiş örnek problem fonksiyonu
def sphere_function(x):
    """
    Küresel test fonksiyonu: f(x) = x1^2 + x2^2 + ... + xn^2
    Minimum değeri 0'dır ve (0,0,...,0) noktasında elde edilir.
    """
    return np.sum(x ** 2)


# 2D grid için özel fonksiyon
def sphere_function_2d(X1, X2):
    """
    2D grid için küresel fonksiyon
    """
    return X1 ** 2 + X2 ** 2


# Ana program
if __name__ == "__main__":
    # Problem parametreleri
    bounds = [(-5, 5), (-5, 5)]  # x1 ve x2 için sınırlar
    colony_size = 6  # Toplam arı sayısı
    max_iter = 100  # Maksimum iterasyon sayısı

    # ABC algoritmasını oluştur ve çalıştır
    abc = ArtificialBeeColony(sphere_function, bounds, colony_size, max_iter)
    best_solution, best_value = abc.optimize()

    # Yakınsama grafiğini çiz
    plt.figure(figsize=(10, 6))
    plt.plot(abc.convergence_curve, 'b-', linewidth=2, label='ABC Algoritması')
    plt.title('ABC Algoritması - Yakınsama Grafiği', fontsize=14)
    plt.xlabel('İterasyon', fontsize=12)
    plt.ylabel('En İyi Değer', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Logaritmik ölçekte daha iyi görselleştirme
    plt.legend()
    plt.tight_layout()
    plt.savefig('abc_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 3D grafik çizimi (sadece 2 boyutlu problemler için)
    if len(bounds) == 2:
        # Amaç fonksiyonunu görselleştirme
        x1 = np.linspace(bounds[0][0], bounds[0][1], 100)
        x2 = np.linspace(bounds[1][0], bounds[1][1], 100)
        X1, X2 = np.meshgrid(x1, x2)
        Z = sphere_function_2d(X1, X2)  # Düzeltilmiş fonksiyon kullanılıyor

        fig = plt.figure(figsize=(15, 6))

        # 3D yüzey grafiği
        ax1 = fig.add_subplot(121, projection='3d')
        surf = ax1.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)
        ax1.scatter(best_solution[0], best_solution[1], best_value,
                    color='red', s=200, label='En İyi Çözüm', marker='*')
        ax1.set_xlabel('x1', fontsize=12)
        ax1.set_ylabel('x2', fontsize=12)
        ax1.set_zlabel('f(x)', fontsize=12)
        ax1.set_title('Amaç Fonksiyonu 3D Yüzey Grafiği', fontsize=14)
        ax1.legend()

        # Kontur grafiği
        ax2 = fig.add_subplot(122)
        contour = ax2.contour(X1, X2, Z, levels=20)
        ax2.clabel(contour, inline=True, fontsize=8)
        ax2.scatter(best_solution[0], best_solution[1], color='red', s=200,
                    label='En İyi Çözüm', marker='*')
        ax2.set_xlabel('x1', fontsize=12)
        ax2.set_ylabel('x2', fontsize=12)
        ax2.set_title('Kontur Grafiği', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('abc_solution_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()

    # Detaylı sonuç raporu
    print("\n" + "=" * 60)
    print("DETAYLI SONUÇ RAPORU")
    print("=" * 60)
    print(f"Problem: f(x) = x1² + x2², -5 ≤ x1, x2 ≤ 5")
    print(f"Beklenen Optimum Çözüm: [0, 0]")
    print(f"Beklenen Optimum Değer: 0")
    print(f"ABC ile Bulunan Çözüm: [{best_solution[0]:.6f}, {best_solution[1]:.6f}]")
    print(f"ABC ile Bulunan Değer: {best_value:.10f}")
    print(f"Hata Oranı: {abs(best_value - 0):.10f}")
    print(f"Toplam İterasyon: {max_iter}")
    print(f"Toplam Çalışma Süresi: {abc.execution_time:.6f} saniye")
    print(f"Son Iterasyondaki En İyi Fitness: {abc.convergence_curve[-1]:.10f}")

    # Başlangıç ve son durum karşılaştırması
    print("\nALGORİTMA PERFORMANSI:")
    print(f"- İlk 10 iterasyonda en iyi değer: {abc.convergence_curve[9]:.6f}")
    print(f"- Son 10 iterasyonda en iyi değer: {abc.convergence_curve[-1]:.10f}")
    print(
        f"- İyileşme Oranı: {((abc.convergence_curve[0] - abc.convergence_curve[-1]) / abc.convergence_curve[0]) * 100:.2f}%")