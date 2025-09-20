"""
Temas de Seguridad en Internet
Versión: 0.6.0
Base de conocimientos sobre ciberseguridad para entrenamiento de IA
"""

from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum

class SecurityLevel(Enum):
    """Niveles de dificultad en seguridad"""
    BASICO = "básico"
    INTERMEDIO = "intermedio"
    AVANZADO = "avanzado"
    EXPERTO = "experto"

class TopicCategory(Enum):
    """Categorías de temas de seguridad"""
    CONCEPTOS_BASICOS = "conceptos_basicos"
    AMENAZAS = "amenazas"
    DEFENSAS = "defensas"
    HERRAMIENTAS = "herramientas"
    LEGISLACION = "legislacion"
    MEJORES_PRACTICAS = "mejores_practicas"
    CODIGO_SEGURO = "codigo_seguro"

@dataclass
class SecurityTopic:
    """Estructura de un tema de seguridad"""
    id: str
    title: str
    category: TopicCategory
    level: SecurityLevel
    description: str
    keywords: List[str]
    learning_objectives: List[str]
    practical_examples: List[str]
    code_examples: List[str]
    resources: List[str]

class SecurityTopics:
    """Base de conocimientos sobre seguridad en internet"""
    
    def __init__(self):
        self.topics = self._initialize_topics()
    
    def _initialize_topics(self) -> Dict[str, SecurityTopic]:
        """Inicializa todos los temas de seguridad"""
        topics = {}
        
        # CONCEPTOS BÁSICOS
        topics["autenticacion"] = SecurityTopic(
            id="autenticacion",
            title="Autenticación y Autorización",
            category=TopicCategory.CONCEPTOS_BASICOS,
            level=SecurityLevel.BASICO,
            description="Fundamentos de autenticación y autorización en sistemas informáticos",
            keywords=["autenticación", "autorización", "login", "password", "2FA", "MFA"],
            learning_objectives=[
                "Entender la diferencia entre autenticación y autorización",
                "Implementar sistemas de autenticación seguros",
                "Conocer métodos de autenticación multifactor",
                "Aplicar principios de least privilege"
            ],
            practical_examples=[
                "Sistema de login con hash de contraseñas",
                "Implementación de 2FA con TOTP",
                "Control de acceso basado en roles (RBAC)",
                "Sesiones seguras con JWT"
            ],
            code_examples=[
                "Hash de contraseñas con bcrypt",
                "Validación de tokens JWT",
                "Middleware de autenticación",
                "Encriptación de datos sensibles"
            ],
            resources=[
                "OWASP Authentication Cheat Sheet",
                "NIST Guidelines for Authentication",
                "RFC 6749 - OAuth 2.0"
            ]
        )
        
        topics["encriptacion"] = SecurityTopic(
            id="encriptacion",
            title="Encriptación y Criptografía",
            category=TopicCategory.CONCEPTOS_BASICOS,
            level=SecurityLevel.INTERMEDIO,
            description="Técnicas de encriptación para proteger datos en tránsito y en reposo",
            keywords=["encriptación", "criptografía", "AES", "RSA", "SSL", "TLS", "hash"],
            learning_objectives=[
                "Entender algoritmos de encriptación simétrica y asimétrica",
                "Implementar encriptación de datos sensibles",
                "Configurar comunicaciones seguras (HTTPS/TLS)",
                "Manejar claves criptográficas de forma segura"
            ],
            practical_examples=[
                "Encriptación de base de datos",
                "Comunicación segura cliente-servidor",
                "Firma digital de documentos",
                "Almacenamiento seguro de contraseñas"
            ],
            code_examples=[
                "Implementación AES-256",
                "Generación de claves RSA",
                "Certificados SSL/TLS",
                "Funciones hash seguras"
            ],
            resources=[
                "Cryptographic Right Answers",
                "OWASP Cryptographic Storage",
                "NIST Cryptographic Standards"
            ]
        )
        
        # AMENAZAS
        topics["malware"] = SecurityTopic(
            id="malware",
            title="Malware y Amenazas Maliciosas",
            category=TopicCategory.AMENAZAS,
            level=SecurityLevel.INTERMEDIO,
            description="Tipos de malware y técnicas de detección y prevención",
            keywords=["malware", "virus", "trojan", "ransomware", "spyware", "rootkit"],
            learning_objectives=[
                "Identificar diferentes tipos de malware",
                "Implementar sistemas de detección",
                "Desarrollar estrategias de prevención",
                "Crear herramientas de análisis forense"
            ],
            practical_examples=[
                "Sistema de detección de malware",
                "Análisis de comportamiento sospechoso",
                "Sandboxing de archivos",
                "Monitoreo de red en tiempo real"
            ],
            code_examples=[
                "Scanner de archivos",
                "Análisis de firmas",
                "Detección heurística",
                "Sistema de cuarentena"
            ],
            resources=[
                "MITRE ATT&CK Framework",
                "Malware Analysis Techniques",
                "YARA Rules Development"
            ]
        )
        
        topics["phishing"] = SecurityTopic(
            id="phishing",
            title="Phishing y Ingeniería Social",
            category=TopicCategory.AMENAZAS,
            level=SecurityLevel.BASICO,
            description="Técnicas de phishing y cómo combatirlas",
            keywords=["phishing", "spear phishing", "whaling", "social engineering", "spoofing"],
            learning_objectives=[
                "Reconocer ataques de phishing",
                "Implementar filtros anti-phishing",
                "Educar usuarios sobre amenazas",
                "Desarrollar sistemas de detección"
            ],
            practical_examples=[
                "Filtro de emails maliciosos",
                "Detección de URLs sospechosas",
                "Sistema de alertas",
                "Simulaciones de phishing"
            ],
            code_examples=[
                "Parser de emails",
                "Validación de URLs",
                "Análisis de contenido",
                "Sistema de scoring"
            ],
            resources=[
                "Anti-Phishing Working Group",
                "Phishing Detection Techniques",
                "User Awareness Training"
            ]
        )
        
        # DEFENSAS
        topics["firewall"] = SecurityTopic(
            id="firewall",
            title="Firewalls y Filtrado de Red",
            category=TopicCategory.DEFENSAS,
            level=SecurityLevel.INTERMEDIO,
            description="Implementación y configuración de firewalls",
            keywords=["firewall", "iptables", "nftables", "ACL", "NAT", "DMZ"],
            learning_objectives=[
                "Configurar firewalls de red y host",
                "Implementar reglas de filtrado",
                "Monitorear tráfico de red",
                "Responder a incidentes de seguridad"
            ],
            practical_examples=[
                "Firewall con iptables",
                "Filtrado de tráfico malicioso",
                "Configuración de DMZ",
                "Monitoreo de conexiones"
            ],
            code_examples=[
                "Scripts de iptables",
                "Parser de logs de firewall",
                "Sistema de alertas",
                "Configuración automática"
            ],
            resources=[
                "iptables Documentation",
                "Firewall Best Practices",
                "Network Security Monitoring"
            ]
        )
        
        topics["ids_ips"] = SecurityTopic(
            id="ids_ips",
            title="Sistemas de Detección de Intrusiones (IDS/IPS)",
            category=TopicCategory.DEFENSAS,
            level=SecurityLevel.AVANZADO,
            description="Implementación de sistemas IDS/IPS",
            keywords=["IDS", "IPS", "Snort", "Suricata", "SIEM", "anomaly detection"],
            learning_objectives=[
                "Implementar sistemas IDS/IPS",
                "Desarrollar reglas de detección",
                "Analizar patrones de ataque",
                "Integrar con sistemas SIEM"
            ],
            practical_examples=[
                "Implementación de Snort",
                "Detección de ataques DDoS",
                "Análisis de tráfico de red",
                "Sistema de correlación de eventos"
            ],
            code_examples=[
                "Parser de paquetes de red",
                "Motor de reglas",
                "Sistema de alertas",
                "API de integración"
            ],
            resources=[
                "Snort User Manual",
                "Suricata Documentation",
                "SIEM Implementation Guide"
            ]
        )
        
        # HERRAMIENTAS
        topics["vulnerability_assessment"] = SecurityTopic(
            id="vulnerability_assessment",
            title="Evaluación de Vulnerabilidades",
            category=TopicCategory.HERRAMIENTAS,
            level=SecurityLevel.INTERMEDIO,
            description="Herramientas y técnicas para evaluar vulnerabilidades",
            keywords=["vulnerability", "scanner", "Nessus", "OpenVAS", "penetration testing"],
            learning_objectives=[
                "Realizar evaluaciones de vulnerabilidades",
                "Interpretar resultados de scanners",
                "Priorizar vulnerabilidades",
                "Desarrollar planes de remediación"
            ],
            practical_examples=[
                "Scanner de puertos",
                "Detección de servicios",
                "Análisis de vulnerabilidades",
                "Reportes automatizados"
            ],
            code_examples=[
                "Scanner de puertos TCP/UDP",
                "Detector de servicios",
                "Parser de CVE",
                "Generador de reportes"
            ],
            resources=[
                "NIST Vulnerability Database",
                "CVE Database",
                "OWASP Testing Guide"
            ]
        )
        
        # CÓDIGO SEGURO
        topics["secure_coding"] = SecurityTopic(
            id="secure_coding",
            title="Desarrollo de Código Seguro",
            category=TopicCategory.CODIGO_SEGURO,
            level=SecurityLevel.INTERMEDIO,
            description="Principios y prácticas para desarrollar código seguro",
            keywords=["secure coding", "OWASP", "input validation", "output encoding", "error handling"],
            learning_objectives=[
                "Aplicar principios de código seguro",
                "Implementar validación de entrada",
                "Prevenir vulnerabilidades comunes",
                "Realizar code review de seguridad"
            ],
            practical_examples=[
                "Validación de entrada de usuario",
                "Sanitización de datos",
                "Manejo seguro de errores",
                "Logging de seguridad"
            ],
            code_examples=[
                "Validación de formularios",
                "Escape de caracteres",
                "Manejo de excepciones",
                "Auditoría de código"
            ],
            resources=[
                "OWASP Secure Coding Practices",
                "CERT Secure Coding Standards",
                "SANS Secure Coding"
            ]
        )
        
        topics["web_security"] = SecurityTopic(
            id="web_security",
            title="Seguridad en Aplicaciones Web",
            category=TopicCategory.CODIGO_SEGURO,
            level=SecurityLevel.INTERMEDIO,
            description="Protección de aplicaciones web contra vulnerabilidades comunes",
            keywords=["XSS", "CSRF", "SQL injection", "OWASP Top 10", "CORS", "CSP"],
            learning_objectives=[
                "Prevenir vulnerabilidades OWASP Top 10",
                "Implementar controles de seguridad web",
                "Configurar headers de seguridad",
                "Realizar testing de seguridad web"
            ],
            practical_examples=[
                "Prevención de XSS",
                "Protección contra CSRF",
                "Validación de SQL injection",
                "Configuración de CORS"
            ],
            code_examples=[
                "Escape de HTML",
                "Tokens CSRF",
                "Prepared statements",
                "Headers de seguridad"
            ],
            resources=[
                "OWASP Top 10",
                "Web Security Testing Guide",
                "Mozilla Security Guidelines"
            ]
        )
        
        # LEGISLACIÓN
        topics["gdpr"] = SecurityTopic(
            id="gdpr",
            title="GDPR y Protección de Datos",
            category=TopicCategory.LEGISLACION,
            level=SecurityLevel.INTERMEDIO,
            description="Cumplimiento del GDPR y protección de datos personales",
            keywords=["GDPR", "privacy", "data protection", "consent", "right to be forgotten"],
            learning_objectives=[
                "Entender requisitos del GDPR",
                "Implementar medidas de privacidad",
                "Desarrollar sistemas de consentimiento",
                "Gestionar derechos de los usuarios"
            ],
            practical_examples=[
                "Sistema de consentimiento",
                "Anonimización de datos",
                "Auditoría de acceso",
                "Portabilidad de datos"
            ],
            code_examples=[
                "Encriptación de datos personales",
                "Sistema de logging de acceso",
                "API de exportación de datos",
                "Mecanismo de borrado"
            ],
            resources=[
                "GDPR Official Text",
                "ICO Guidance",
                "Privacy by Design"
            ]
        )
        
        return topics
    
    def get_topic(self, topic_id: str) -> SecurityTopic:
        """Obtiene un tema específico"""
        return self.topics.get(topic_id)
    
    def get_topics_by_category(self, category: TopicCategory) -> List[SecurityTopic]:
        """Obtiene temas por categoría"""
        return [topic for topic in self.topics.values() if topic.category == category]
    
    def get_topics_by_level(self, level: SecurityLevel) -> List[SecurityTopic]:
        """Obtiene temas por nivel de dificultad"""
        return [topic for topic in self.topics.values() if topic.level == level]
    
    def get_all_topics(self) -> List[SecurityTopic]:
        """Obtiene todos los temas"""
        return list(self.topics.values())
    
    def search_topics(self, query: str) -> List[SecurityTopic]:
        """Busca temas por palabras clave"""
        query_lower = query.lower()
        results = []
        
        for topic in self.topics.values():
            if (query_lower in topic.title.lower() or
                query_lower in topic.description.lower() or
                any(query_lower in keyword.lower() for keyword in topic.keywords)):
                results.append(topic)
        
        return results
    
    def get_learning_path(self, start_level: SecurityLevel = SecurityLevel.BASICO) -> List[SecurityTopic]:
        """Obtiene una ruta de aprendizaje progresiva"""
        levels = [SecurityLevel.BASICO, SecurityLevel.INTERMEDIO, SecurityLevel.AVANZADO, SecurityLevel.EXPERTO]
        start_index = levels.index(start_level)
        
        learning_path = []
        for level in levels[start_index:]:
            level_topics = self.get_topics_by_level(level)
            learning_path.extend(level_topics)
        
        return learning_path
    
    def get_topic_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas de los temas"""
        total_topics = len(self.topics)
        
        by_category = {}
        for category in TopicCategory:
            by_category[category.value] = len(self.get_topics_by_category(category))
        
        by_level = {}
        for level in SecurityLevel:
            by_level[level.value] = len(self.get_topics_by_level(level))
        
        return {
            "total_topics": total_topics,
            "by_category": by_category,
            "by_level": by_level
        }
